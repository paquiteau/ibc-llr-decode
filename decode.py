"""Decoding pipeline: load per-trial GLM maps and run cross-validated SVM decoding."""

import contextlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from nilearn.image import concat_imgs, math_img
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.maskers import NiftiMasker
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    cross_validate,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from glm import find_brain_mask


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report progress into a tqdm bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    setattr(joblib.parallel, "BatchCompletionCallBack", TqdmBatchCompletionCallback)
    try:
        yield tqdm_object
    finally:
        setattr(joblib.parallel, "BatchCompletionCallBack", old)
        tqdm_object.close()


def _load_one_file(file, mask, dtype=np.float32):
    # Downcast right after transform(): the source NIfTI is already float32
    # on disk (its header dtype), so nilearn's get_fdata() float64 upcast
    # inside mask.transform() adds no real precision — only doubles memory
    # and the BLAS cost of every downstream ANOVA/SVM fit.
    beta_data = mask.transform(file).astype(dtype, copy=False)
    entities = parse_bids_filename(file)["entities"]
    sub, cond, run = entities["sub"], entities["cond"], entities["run"]
    return [cond] * beta_data.shape[0], [f"{sub}-{run}"] * beta_data.shape[0], beta_data


def _cache_path(cache_dir, derivatives_folder, sub_label):
    return Path(cache_dir) / Path(derivatives_folder).name / f"sub-{sub_label}.npz"


def _load_cached(cache_path):
    """Return (X, y, g) from `cache_path`, or None on a miss/unreadable file.

    Never raises — a corrupt or half-written cache file (e.g. from a crashed
    process) just falls back to recomputing, same as a cold cache.
    """
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as data:
            return data["X"], data["y"], data["g"]
    except Exception:
        return None


def load_cached_glm_data(cache_dir, derivatives_folder, sub_label):
    """Return cached (X, y, g) for one subject, written by `load_glm_data`, or None.

    Lets a backfill (e.g. adding Haufe pattern maps to an existing `decode`
    run) reuse the already-parsed beta/z-score arrays without touching the
    source NIfTIs, as long as that subject/strategy was previously loaded
    with a matching `cache_dir`.
    """
    return _load_cached(_cache_path(cache_dir, derivatives_folder, sub_label))


def _save_cache_atomic(cache_path, X, y, g):
    """Write (X, y, g) to `cache_path` via a same-dir temp file + atomic rename.

    Safe under concurrent writers targeting the same path (several parallel
    jobs populating a cold cache for the same subject/strategy): os.replace
    is an atomic rename on POSIX, so the loser's write is simply overwritten
    — never a torn/partial file for whoever reads it next.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp_name, "wb") as fh:
            np.savez(fh, X=X, y=y, g=g)
        os.replace(tmp_name, cache_path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def load_glm_data(
    dataset_dir,
    derivatives_folder,
    mask_derivatives_folder,
    sub_labels,
    task_label=None,
    glm_folder="glm",
    suffix="zscore",
    n_jobs=10,
    cache_dir=None,
):
    """Load LSS GLM beta/z-score maps for one or more subjects.

    Returns (X, y, groups, masker): stacked beta/z-score arrays (float32 — see
    `_load_one_file`), condition labels, "<sub>-<run>" group labels (used for
    leave-run-out cross-validation), and the fitted NiftiMasker — needed to
    map SVM weight vectors back to brain space. The masker is only meaningful
    when `sub_labels` has a single subject (the only way this is currently
    called); with several subjects it is just the last one's mask.

    `cache_dir`, if given, caches the loaded (X, y, g) per
    `(Path(derivatives_folder).name, sub_label)` — independent of any
    downstream k/p/tag — so repeated calls for the same subject/strategy
    (e.g. sweeping k) skip re-parsing NIfTI files entirely. The masker itself
    is not cached; rebuilding it is cheap. `None` (default) disables caching,
    unchanged from prior behavior.
    """
    all_events, all_group_labels, all_beta_maps = [], [], []

    # Path(dataset_dir) / derivatives_folder — not string concatenation — so an
    # absolute derivatives_folder correctly overrides dataset_dir, matching the
    # convention used everywhere else (glm.py, find_brain_mask, denoise.py).
    search_root = Path(dataset_dir) / derivatives_folder

    for sub_label in sub_labels:
        mask_path = find_brain_mask(
            dataset_dir, sub_label, fmriprep_folder=mask_derivatives_folder
        )
        mask = NiftiMasker(mask_img=mask_path).fit()

        cached = None
        if cache_dir is not None:
            cache_path = _cache_path(cache_dir, derivatives_folder, sub_label)
            cached = _load_cached(cache_path)

        if cached is not None:
            beta_maps, events, group_labels = cached
        else:
            task_glob = f"*task-{task_label}*" if task_label else "*"
            pattern = (
                f"sub-{sub_label}/ses-*/{glm_folder}/"
                f"sub-{sub_label}_ses-*{task_glob}_cond-*_{suffix}.nii.gz"
            )
            files = sorted(str(p) for p in search_root.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No files found for sub-{sub_label} in "
                    f"{search_root} with suffix '{suffix}'.\n"
                    f"Check --glm-folder (current: {glm_folder}) and --suffix."
                )

            with tqdm_joblib(tqdm(desc=f"Loading sub-{sub_label}", total=len(files))):
                res = Parallel(n_jobs=n_jobs)(
                    delayed(_load_one_file)(f, mask) for f in files
                )

            res = list(map(list, zip(*res)))
            events = np.concatenate(res[0], axis=0)
            group_labels = np.concatenate(res[1], axis=0)
            beta_maps = np.concatenate(res[2], axis=0)

            if cache_dir is not None:
                _save_cache_atomic(cache_path, beta_maps, events, group_labels)

        all_events.append(events)
        all_group_labels.append(group_labels)
        all_beta_maps.append(beta_maps)

    return (
        np.concatenate(all_beta_maps, axis=0),
        np.concatenate(all_events, axis=0),
        np.concatenate(all_group_labels, axis=0),
        mask,
    )


def _make_pipeline(k=1000):
    """Build the OvR SVM pipeline (ANOVA feature selection + LinearSVC)."""
    return OneVsRestClassifier(
        Pipeline(
            [
                ("anova", SelectKBest(f_classif, k=k)),
                ("svc", LinearSVC(max_iter=10000)),
            ]
        )
    )


def run_decoding(X, y, g, n_jobs=6, k=1500, nested=False, param_grid=None):
    """Cross-validated SVM decoding with LeaveOneGroupOut.

    Two modes, controlled by `nested`:

    nested=False (default) — single-loop CV, fixed hyperparameters
        (`k` for SelectKBest, LinearSVC's default C=1.0). No tuning. Fast.

    nested=True — nested (double) cross-validation:
        Outer loop: LeaveOneGroupOut over runs — one run held out purely
        for evaluation, never touched during hyperparameter selection.
        Inner loop: LeaveOneGroupOut over the *remaining* runs in each
        outer training fold — used only to pick the best (k, C) combo for
        that fold, via GridSearchCV. After the inner search, GridSearchCV
        refits automatically on the *full* outer training set with the
        winning hyperparameters, so the final model evaluated on the outer
        test run is trained on exactly as much data as in the non-nested
        mode. The inner loop only ever sees fewer runs *during the search
        itself*. `param_grid` lets you override the default grid, e.g.:
            {"estimator__anova__k": [5000, 20000], "estimator__svc__C": [0.1, 1]}

    Returns
    -------
    cv_results : dict
        sklearn cross_validate output. In nested mode, `cv_results["estimator"]`
        holds one fitted GridSearchCV object per outer fold (not a plain SVM).
    cv_predict : np.ndarray
        Out-of-fold predictions, one per trial.
    best_params_per_fold : list[dict] | None
        Only in nested mode: the winning (k, C) per outer fold — check these
        for stability across folds/runs. `None` in non-nested mode.
    fold_sizes : list[tuple[int, int]]
        (n_train, n_test) trial counts for each outer fold — needed to
        correctly parameterize the Nadeau-Bengio corrected t-test downstream.
    """
    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups=g))
    fold_sizes = [
        (len(train_idx), len(test_idx)) for train_idx, test_idx in outer_splits
    ]

    if not nested:
        pipeline = _make_pipeline(k=k)
        cv_results = cross_validate(
            clone(pipeline),
            X,
            y,
            cv=outer_splits,
            groups=g,
            verbose=0,
            n_jobs=n_jobs,
            return_estimator=True,
        )
        best_params_per_fold = None
    else:
        if param_grid is None:
            param_grid = {
                "estimator__anova__k": [5000, 20000, 50000],
                "estimator__svc__C": [0.01, 0.1, 1, 10],
            }

        pipeline = _make_pipeline()
        inner_cv = LeaveOneGroupOut()
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            refit=True,
        )

        # `groups` is passed as a fit_param so sklearn slices it to the outer
        # training indices for each fold, and forwards it to GridSearchCV.fit(),
        # which uses it for its own inner LeaveOneGroupOut split.
        cv_results = cross_validate(
            grid,
            X,
            y,
            cv=outer_splits,
            groups=g,
            fit_params={"groups": g},
            return_estimator=True,
            n_jobs=1,  # outer loop sequential: inner GridSearchCV already parallelizes
            verbose=1,
        )
        best_params_per_fold = [est.best_params_ for est in cv_results["estimator"]]

    # Derive predictions from saved estimators — no second CV pass
    cv_predict = np.empty_like(y)
    for est, (_, test_idx) in zip(cv_results["estimator"], outer_splits):
        cv_predict[test_idx] = est.predict(X[test_idx])

    return cv_results, cv_predict, best_params_per_fold, fold_sizes


def compare_group_sizes(X, y, g, group_sizes=(1, 2, 3), k=1500, n_jobs=6):
    """Compare LeavePGroupsOut(p) for different held-out sizes p, on the same data.

    Not a hyperparameter search: nothing is tuned here, this only checks how
    stable the estimated performance is depending on how many runs are left
    out at each outer fold (p=1, 2, 3...).

    Returns: tuple (results_dict, predictions_dict), keyed by p.
    """
    results = {}
    predictions = {}
    for p in group_sizes:
        cv = LeavePGroupsOut(n_groups=p)
        splits = list(cv.split(X, y, groups=g))

        pipeline = _make_pipeline(k=k)
        cv_results = cross_validate(
            clone(pipeline),
            X,
            y,
            cv=splits,
            groups=g,
            n_jobs=n_jobs,
            return_estimator=True,
        )
        scores = cv_results["test_score"]
        results[p] = scores

        cv_predict = np.empty_like(y)
        for est, (_, test_idx) in zip(cv_results["estimator"], splits):
            cv_predict[test_idx] = est.predict(X[test_idx])
        predictions[p] = cv_predict

        print(
            f"p={p} ({len(splits)} folds) : {scores.mean():.3f} +/- {scores.std():.3f}"
        )

    return results, predictions


def save_group_size_comparison(
    output_dir, sub_label, results, k, predictions=None, y=None, tag="decode"
):
    condition = f"test_anova_{k}"
    out = Path(output_dir) / f"sub-{sub_label}" / tag / condition
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("scores_p*.npy"):
        stale.unlink()
    summary_path = out / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    summary = {
        str(p): {"mean": float(s.mean()), "std": float(s.std()), "n_folds": len(s)}
        for p, s in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    for p, scores in results.items():
        np.save(out / f"scores_p{p}.npy", scores)

    if predictions is not None:
        for p, preds in predictions.items():
            np.save(out / f"cv_predict_p{p}.npy", preds)

    if y is not None:
        np.save(out / "y.npy", y)

    return out


def save_results(
    output_dir,
    sub_label,
    tag,
    cv_results,
    cv_predict,
    y,
    g,
    k,
    best_params_per_fold=None,
    fold_sizes=None,
):
    condition = f"test_anova_{k}"
    out = Path(output_dir) / f"sub-{sub_label}" / tag / condition
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("svm_fold_*.pkl"):
        stale.unlink()
    for fold_idx, est in enumerate(cv_results["estimator"]):
        model_to_save = est.best_estimator_ if hasattr(est, "best_estimator_") else est
        joblib.dump(model_to_save, out / f"svm_fold_{fold_idx}.pkl")
    if best_params_per_fold is not None:
        with open(out / "best_params_per_fold.json", "w") as f:
            json.dump(best_params_per_fold, f, indent=2)
    if fold_sizes is not None:
        with open(out / "fold_sizes.json", "w") as f:
            json.dump(fold_sizes, f, indent=2)
    np.save(out / "cvscores.npy", cv_results["test_score"])
    np.save(out / "cv_predict.npy", cv_predict)
    np.save(out / "y.npy", y)
    np.save(out / "g.npy", g)
    return out


def load_saved_models(decode_dir, sub_label, tag, k):
    """Load the svm_fold_*.pkl models written by save_results(), in fold order."""
    condition = f"test_anova_{k}"
    out = Path(decode_dir) / f"sub-{sub_label}" / tag / condition
    paths = sorted(
        out.glob("svm_fold_*.pkl"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
    )
    return [joblib.load(p) for p in paths]


def average_weight_maps(models, masker):
    """Average per-class SVM weight vectors across CV folds, mapped back to brain space.

    Each model is a fitted OneVsRestClassifier(Pipeline([anova, svc])) — one per
    outer CV fold (or a GridSearchCV wrapping one, in nested mode, unwrapped via
    `best_estimator_` same as `save_results`). SelectKBest can pick different
    voxels per fold; a voxel not selected in a given fold contributes zero to
    that fold's average, which is the standard way to pool weight maps across
    folds with per-fold feature selection.

    Returns dict {class_label: nibabel Nifti1Image}.
    """
    models = [
        est.best_estimator_ if hasattr(est, "best_estimator_") else est
        for est in models
    ]
    classes = models[0].classes_
    n_features = len(models[0].estimators_[0].named_steps["anova"].get_support())

    weight_maps = {}
    for class_idx, label in enumerate(classes):
        acc = np.zeros(n_features)
        for model in models:
            est = model.estimators_[class_idx]
            support = est.named_steps["anova"].get_support()
            acc[support] += est.named_steps["svc"].coef_.ravel()
        weight_maps[label] = masker.inverse_transform(acc / len(models))
    return weight_maps


def average_pattern_maps(models, X, g, masker):
    """Haufe-transformed activation patterns, averaged across CV folds.

    Raw SVM weights (`average_weight_maps`) can give a voxel a large weight
    purely to cancel noise correlated with the true signal in other voxels —
    they tell you what the classifier relies on, not where the discriminative
    signal actually lives. Activation patterns (Haufe et al., 2014,
    NeuroImage) fix this by measuring how each voxel covaries with the
    classifier's decision variable instead: A = Cov(X) @ W. Computed here per
    fold as A_j = Cov(x_j, s), with s = X_support @ w, on that fold's
    training trials (the same trials the weight vector was fit on) — this is
    algebraically equivalent to the full Cov(X) @ W but avoids ever forming
    the (n_selected x n_selected) covariance matrix.

    `X`, `g` must be the same arrays passed to `run_decoding` — fold order
    is recovered by replaying the same LeaveOneGroupOut split.

    Returns dict {class_label: nibabel Nifti1Image}, same shape as
    `average_weight_maps`.
    """
    models = [
        est.best_estimator_ if hasattr(est, "best_estimator_") else est
        for est in models
    ]
    outer_splits = list(LeaveOneGroupOut().split(X, groups=g))
    classes = models[0].classes_
    n_features = len(models[0].estimators_[0].named_steps["anova"].get_support())

    pattern_maps = {}
    for class_idx, label in enumerate(classes):
        acc = np.zeros(n_features)
        for model, (train_idx, _) in zip(models, outer_splits):
            est = model.estimators_[class_idx]
            support = est.named_steps["anova"].get_support()
            w = est.named_steps["svc"].coef_.ravel()
            X_support = X[train_idx][:, support]
            s = X_support @ w
            Xc = X_support - X_support.mean(axis=0)
            sc = s - s.mean()
            acc[support] += Xc.T @ sc / (len(sc) - 1)
        pattern_maps[label] = masker.inverse_transform(acc / len(models))
    return pattern_maps


def collapse_norm_map(maps):
    """L2 norm across classes: one "how important is this voxel" map per subject.

    `maps` : dict {class_label: 3D Nifti1Image}, from `average_weight_maps`
    or `average_pattern_maps`.
    """
    labels = sorted(maps)
    stacked = concat_imgs([maps[label] for label in labels])
    return math_img("np.linalg.norm(img, axis=-1)", img=stacked)


def save_weight_maps(output_dir, sub_label, tag, k, weight_maps, kind="weight"):
    """Save all classes' weight/pattern maps as one 4D NIfTI + a label sidecar.

    `weight_maps` : dict {class_label: 3D Nifti1Image}, from
    `average_weight_maps()` (kind="weight") or `average_pattern_maps()`
    (kind="pattern"). Written as `<kind>_maps.nii.gz` (volume order =
    alphabetical label order) plus `<kind>_maps_labels.json` (that same label
    list) — one 4D file instead of one per class is simpler to keep track of,
    and NiftiMasker.inverse_transform()'s output only carries an sform
    (qform_code left at 0), which some viewers ignore when overlaying onto
    another image (e.g. the T1w) that has both set — so both are set
    explicitly here to the same affine.
    """
    condition = f"test_anova_{k}"
    out = Path(output_dir) / f"sub-{sub_label}" / tag / condition
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(f"{kind}_map_*.nii.gz"):
        stale.unlink()

    labels = sorted(weight_maps)
    affine = weight_maps[labels[0]].affine
    data_4d = np.stack([weight_maps[label].get_fdata() for label in labels], axis=-1)
    img_4d = nib.Nifti1Image(data_4d, affine)
    img_4d.set_sform(affine, code=2)
    img_4d.set_qform(affine, code=2)
    img_4d.to_filename(out / f"{kind}_maps.nii.gz")
    (out / f"{kind}_maps_labels.json").write_text(json.dumps(labels))
    return out


def save_norm_map(output_dir, sub_label, tag, k, norm_img, kind="weight"):
    """Save a norm-collapsed map (see `collapse_norm_map`) as `<kind>_map_norm.nii.gz`."""
    condition = f"test_anova_{k}"
    out = Path(output_dir) / f"sub-{sub_label}" / tag / condition
    out.mkdir(parents=True, exist_ok=True)
    norm_img.to_filename(out / f"{kind}_map_norm.nii.gz")
    return out
