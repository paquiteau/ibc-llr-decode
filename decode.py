"""Decoding pipeline: load per-trial GLM maps and run cross-validated SVM decoding."""

import contextlib
import json
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
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


def _load_one_file(file, mask):
    beta_data = mask.transform(file)
    entities = parse_bids_filename(file)["entities"]
    sub, cond, run = entities["sub"], entities["cond"], entities["run"]
    return [cond] * beta_data.shape[0], [f"{sub}-{run}"] * beta_data.shape[0], beta_data


def load_glm_data(
    dataset_dir,
    derivatives_folder,
    mask_derivatives_folder,
    sub_labels,
    task_label=None,
    glm_folder="glm",
    suffix="zscore",
    n_jobs=10,
):
    """Load LSS GLM beta/z-score maps for one or more subjects.

    Returns (X, y, groups): stacked beta/z-score arrays, condition labels, and
    "<sub>-<run>" group labels (used for leave-run-out cross-validation).
    """
    all_events, all_group_labels, all_beta_maps = [], [], []

    # Path(dataset_dir) / derivatives_folder — not string concatenation — so an
    # absolute derivatives_folder correctly overrides dataset_dir, matching the
    # convention used everywhere else (glm.py, find_brain_mask, denoise.py).
    search_root = Path(dataset_dir) / derivatives_folder

    for sub_label in sub_labels:
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

        mask_path = find_brain_mask(
            dataset_dir, sub_label, fmriprep_folder=mask_derivatives_folder
        )
        mask = NiftiMasker(mask_img=mask_path).fit()
        with tqdm_joblib(tqdm(desc=f"Loading sub-{sub_label}", total=len(files))):
            res = Parallel(n_jobs=n_jobs)(
                delayed(_load_one_file)(f, mask) for f in files
            )

        res = list(map(list, zip(*res)))
        all_events.append(np.concatenate(res[0], axis=0))
        all_group_labels.append(np.concatenate(res[1], axis=0))
        all_beta_maps.append(np.concatenate(res[2], axis=0))

    return (
        np.concatenate(all_beta_maps, axis=0),
        np.concatenate(all_events, axis=0),
        np.concatenate(all_group_labels, axis=0),
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
