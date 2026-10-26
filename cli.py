"""Cyclopts CLI for generating the data and figures."""

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from cyclopts.types import ExistingDirectory
from patch_denoise.bindings.cli import parse_dims
from patch_denoise.bindings.utils import fast_cuda_check
from patch_denoise.space_time.base import DenoiserName

from _constants import DATASETS, DEFAULT_SPACE_LABEL, STRATEGIES

app = App()

# Selects the per-dataset defaults of `_constants.DATASETS` (subjects, task,
# TR, ...); explicitly passed options still take precedence.
DatasetName = Literal["ibc", "motif"]

PatchShapeOpt = Annotated[
    tuple[int, ...],
    Parameter(
        negative="",
        converter=parse_dims,
        help='Dims separated by ",". -1 uses the full extent.',
    ),
]
PatchOverlapOpt = Annotated[
    tuple[int, ...],
    Parameter(
        negative="",
        converter=parse_dims,
        help='Dims separated by ",". -1 uses the full extent.',
    ),
]
GpuFlag = Annotated[
    bool,
    Parameter(name="--gpu", negative="--cpu", help="Use GPU or CPU for denoising."),
]


@app.command()
def glm(
    dataset_dir: ExistingDirectory,
    derivatives_folder: Path,
    mask_derivatives_folder: Path,
    task_label: str | None = None,
    desc: str = "preproc",
    tr: float | None = None,
    space_label: str = DEFAULT_SPACE_LABEL,
    sub: list[str] | None = None,
    save_betas: Annotated[bool, Parameter(negative="--no-betas")] = True,
    n_jobs: int = -1,
    glm_folder: str = "glm",
    dataset: DatasetName = "ibc",
):
    """Compute per-trial LSS GLM beta/z-score maps for all subjects and sessions.

    `derivatives_folder`/`mask_derivatives_folder` should be absolute paths
    (e.g. /volatile2/ibc_rsvp/derivatives/fmriprep+denoised) — an absolute path
    here overrides `dataset_dir` in the internal join, which is the simplest way
    to point at a derivatives tree outside the raw dataset directory.
    """
    from glm import make_lss_betas, rsvp_cleanup_events_df

    ds = DATASETS[dataset]
    make_lss_betas(
        dataset_dir,
        task_label or ds.task_label,
        derivatives_folder,
        desc,
        tr or ds.tr,
        mask_derivatives_folder,
        sub_labels=sub,
        space_label=space_label,
        cleanup_func=rsvp_cleanup_events_df,
        save_betas=save_betas,
        n_jobs=n_jobs,
        glm_folder=glm_folder,
    )


@app.command()
def denoise(
    input_dir: ExistingDirectory,
    output_dir: Path,
    task_label: str | None = None,
    sub: list[str] | None = None,
    space_label: str | None = None,
    method: DenoiserName = DenoiserName.OPTIMAL_FRO,
    patch_shape: PatchShapeOpt = (11, 11, 11, -1),
    patch_overlap: PatchOverlapOpt = (5, 5, 5, -1),
    recombination: str = "weighted",
    mask_threshold: int = 50,
    mask_source: str = "auto",
    fmriprep_dir: Path | None = None,
    gpu: GpuFlag = fast_cuda_check(),
    overwrite: bool = False,
    dataset: DatasetName = "ibc",
):
    """Denoise BOLD runs with patch-denoise, writing a mirrored derivatives tree.

    `mask_source="fmriprep"` (with `--fmriprep-dir`) reuses the fMRIPrep brain
    mask — only correct when `input_dir` holds already-fMRIPrepped, space-
    normalized data (pass `--space-label` to match). Use `mask_source="auto"`
    (the default) when denoising raw sourcedata, which computes a per-run EPI
    mask instead.
    """
    from denoise import run_denoising

    run_denoising(
        input_dir,
        output_dir,
        task_label or DATASETS[dataset].task_label,
        sub_labels=sub,
        space_label=space_label,
        method=str(method),
        patch_shape=patch_shape,
        patch_overlap=patch_overlap,
        recombination=recombination,
        mask_threshold=mask_threshold,
        mask_source=mask_source,
        fmriprep_root=fmriprep_dir,
        gpu=gpu,
        overwrite=overwrite,
    )


DEFAULT_ANOVA_SIZE = 10_000


@app.command()
def decode(
    dataset_dir: ExistingDirectory,
    derivatives_folder: Path,
    mask_derivatives_folder: Path,
    output_dir: Path,
    sub: list[str] | None = None,
    task_label: str | None = None,
    glm_folder: str = "glm",
    suffix: str = "zscore",
    n_jobs: int = 1,
    k: int = DEFAULT_ANOVA_SIZE,
    nested: bool = False,
    leave_p_out: list[int] | None = None,
    tag: str = "decode",
    cache_dir: Path | None = None,
    roi_mask: str | None = None,
    dataset: DatasetName = "ibc",
):
    """Fit a leave-p-run-out SVM decoder on LSS beta/z-score maps, per subject.

    `leave_p_out=None` (the default) is the main leave-one-run-out decoding used
    for the strategy comparison in `analysis`. One or more `--leave-p-out` values
    instead run a `LeavePGroupsOut` CV-robustness sweep (how stable/how does
    accuracy change as more runs are held out), saved separately.

    Pass every `--leave-p-out` value you want in ONE call, e.g.
    `--leave-p-out 1 --leave-p-out 2 --leave-p-out 3` — `save_group_size_comparison`
    overwrites a subject's whole sweep output each call, so invoking `decode`
    once per p value would silently discard all but the last one.

    `cache_dir`, if given, caches each subject's loaded beta/z-score arrays
    (keyed by derivatives folder + subject, independent of `k`/`leave_p_out`/
    `tag`) so sweeping multiple `k` values for the same subject/strategy only
    parses the underlying NIfTI files once. Safe to share across concurrent
    `decode` invocations (e.g. a parallel job runner) for different subjects
    or k values — see `decode.load_glm_data`.

    `roi_mask`, if given, restricts decoding to brain mask ∩ ROI (`{sub}` in
    the path is replaced by the subject label); the cache still holds the
    whole-brain arrays, so whole-brain and ROI runs share it. ANOVA still
    selects `k` voxels inside the ROI. Write ROI runs to their own
    `output_dir`.
    """
    from decode import (
        average_pattern_maps,
        average_weight_maps,
        collapse_norm_map,
        compare_group_sizes,
        load_glm_data,
        roi_selection,
        run_decoding,
        save_group_size_comparison,
        save_norm_map,
        save_results,
        save_weight_maps,
    )

    ds = DATASETS[dataset]
    for sub_label in sub or ds.subjects:
        X, y, g, masker = load_glm_data(
            dataset_dir,
            derivatives_folder,
            mask_derivatives_folder,
            task_label=task_label or ds.task_label,
            glm_folder=glm_folder,
            suffix=suffix,
            sub_labels=[sub_label],
            n_jobs=n_jobs,
            cache_dir=cache_dir,
        )
        if roi_mask is not None:
            keep, masker = roi_selection(masker, roi_mask.format(sub=sub_label))
            X = X[:, keep]
        if not leave_p_out:
            cv_results, cv_predict, best_params_per_fold, fold_sizes = run_decoding(
                X,
                y,
                g,
                n_jobs=n_jobs,
                k=k,
                nested=nested,
            )
            save_results(
                output_dir,
                sub_label,
                tag,
                cv_results,
                cv_predict,
                y,
                g,
                k,
                best_params_per_fold,
                fold_sizes,
            )
            weight_maps = average_weight_maps(cv_results["estimator"], masker)
            save_weight_maps(output_dir, sub_label, tag, k, weight_maps, kind="weight")
            save_norm_map(
                output_dir,
                sub_label,
                tag,
                k,
                collapse_norm_map(weight_maps),
                kind="weight",
            )

            pattern_maps = average_pattern_maps(cv_results["estimator"], X, g, masker)
            save_weight_maps(
                output_dir, sub_label, tag, k, pattern_maps, kind="pattern"
            )
            save_norm_map(
                output_dir,
                sub_label,
                tag,
                k,
                collapse_norm_map(pattern_maps),
                kind="pattern",
            )
        else:
            results, predictions, fold_predictions = compare_group_sizes(
                X,
                y,
                g,
                group_sizes=tuple(leave_p_out),
                k=k,
                n_jobs=n_jobs,
            )
            save_group_size_comparison(
                output_dir,
                sub_label,
                results,
                k,
                predictions,
                y,
                tag,
                fold_predictions,
                g,
            )


@app.command()
def weight_maps(
    decode_dir: ExistingDirectory,
    dataset_dir: ExistingDirectory,
    mask_derivatives_folder: Path,
    sub: list[str] | None = None,
    tag: str = "decode",
    k: int = DEFAULT_ANOVA_SIZE,
    cache_dir: Path | None = None,
    derivatives_folder: Path | None = None,
    roi_mask: str | None = None,
    glm_folder: str = "glm",
    dataset: DatasetName = "ibc",
):
    """Backfill/refresh weight/pattern maps for an existing `decode --output-dir` tree.

    Loads the already-fitted svm_fold_*.pkl models saved by `decode` (no
    refitting, no GLM beta reload) and only rebuilds each subject's brain mask
    to map weights back to voxel space. Use this after upgrading `decode.py`
    to add weight maps to decode runs that predate that feature.

    Pass matching `--cache-dir`/`--derivatives-folder` (the same values used
    for that `decode` run) to also backfill Haufe activation-pattern maps
    (`pattern_maps.nii.gz` + `pattern_map_norm.nii.gz`) from the cached
    (X, g) arrays — still no refit, just `average_pattern_maps` on the
    already-fitted models. Without them, only the raw-weight maps are
    (re)written. Pass the `--roi-mask` of that `decode` run, if any.
    """
    from nilearn.maskers import NiftiMasker

    from decode import (
        average_pattern_maps,
        average_weight_maps,
        collapse_norm_map,
        load_cached_glm_data,
        load_saved_models,
        roi_selection,
        save_norm_map,
        save_weight_maps,
    )
    from glm import find_brain_mask

    ds = DATASETS[dataset]
    for sub_label in sub or ds.subjects:
        mask_path = find_brain_mask(
            dataset_dir,
            sub_label,
            fmriprep_folder=mask_derivatives_folder,
            task_label=ds.task_label,
        )
        masker = NiftiMasker(mask_img=mask_path).fit()
        keep = None
        if roi_mask is not None:
            keep, masker = roi_selection(masker, roi_mask.format(sub=sub_label))
        models = load_saved_models(decode_dir, sub_label, tag, k)
        maps = average_weight_maps(models, masker)
        save_weight_maps(decode_dir, sub_label, tag, k, maps, kind="weight")
        save_norm_map(
            decode_dir, sub_label, tag, k, collapse_norm_map(maps), kind="weight"
        )

        if cache_dir is not None and derivatives_folder is not None:
            cached = load_cached_glm_data(
                cache_dir, derivatives_folder, sub_label, glm_folder
            )
            if cached is None:
                print(f"sub-{sub_label}: no cached GLM data, skipping pattern maps")
                continue
            X, _y, g = cached
            if keep is not None:
                X = X[:, keep]
            pattern_maps = average_pattern_maps(models, X, g, masker)
            save_weight_maps(
                decode_dir, sub_label, tag, k, pattern_maps, kind="pattern"
            )
            save_norm_map(
                decode_dir,
                sub_label,
                tag,
                k,
                collapse_norm_map(pattern_maps),
                kind="pattern",
            )


@app.command()
def analysis(
    decode_dir: ExistingDirectory,
    output_dir: Path,
    dataset_dir: ExistingDirectory,
    mask_derivatives_folder: Path,
    strategy: list[str] | None = None,
    sub: list[str] | None = None,
    tag: str = "decode",
    k: int = DEFAULT_ANOVA_SIZE,
    reference_strategy: str = "fmriprep",
    p: list[int] | None = None,
    tag_sweep: str = "cv_sweep",
    dataset: DatasetName = "ibc",
):
    """Group, training-size, single-subject and per-class statistics + figures.

    Every strategy is compared to `reference_strategy` on per-subject paired
    values (exact sign-flip permutation tests, see `analysis`'s module
    docstring and the README).

    `decode_dir` must contain one subdirectory per strategy, each populated by
    a `decode --output-dir <decode_dir>/<strategy>` run (see README). The
    training-size analysis (`--p`, number of runs held out per fold, default
    from `--dataset`) also reads the `--tag cv_sweep --leave-p-out ...` decode
    outputs, and is skipped if they're missing. The single-subject test uses
    the dataset's mini-blocks as units (`_constants.Dataset.block_size`).

    `dataset_dir`/`mask_derivatives_folder` locate each subject's T1w for the
    weight-map figures — always the plain fMRIPrep derivatives (same
    convention as `decode`'s `mask_derivatives_folder`).
    """
    from analysis import run_analysis

    ds = DATASETS[dataset]
    run_analysis(
        decode_dir,
        output_dir,
        dataset_dir,
        mask_derivatives_folder,
        strategies=strategy or list(STRATEGIES),
        subjects=sub or list(ds.subjects),
        tag=tag,
        k=k,
        reference_strategy=reference_strategy,
        ps=p or list(ds.training_ps),
        tag_sweep=tag_sweep,
        block_size=ds.block_size,
    )


@app.command()
def grid_analysis(
    decode_dir: ExistingDirectory,
    output_dir: Path,
    strategy: list[str] | None = None,
    sub: list[str] | None = None,
    k: list[int] | None = None,
    p: list[int] | None = None,
    tag_main: str = "decode",
    tag_sweep: str = "cv_sweep",
    reference_strategy: str = "fmriprep",
):
    """Aggregate a (k, p) grid search into a long dataframe + heatmap/line figures.

    `decode_dir` must contain, per strategy, both a `--tag decode` (k-only)
    tree and a `--tag cv_sweep` (`--leave-p-out 2 3 4 ...`) tree at each `k`
    (see `decode`'s docstring) — p=1 is read from the `decode` tree, never
    from `cv_sweep` (a p=1 cv_sweep run would just duplicate the main decode).

    Answers "which (k, p) region looks good" — a different, exploratory
    question from `analysis`'s "is strategy A better at one fixed, chosen k".
    Run this first to survey, then `analysis --k <chosen>` for the real
    strategy comparison.
    """
    from analysis import run_grid_analysis

    run_grid_analysis(
        decode_dir,
        output_dir,
        strategies=strategy or ["fmriprep", "fmriprep+denoised"],
        subjects=sub or list(DATASETS["ibc"].subjects),
        ks=k or [1500, 3000, 10000, 20000, 50000],
        ps=p or [1, 2, 3, 4, 5],
        tag_main=tag_main,
        tag_sweep=tag_sweep,
        reference_strategy=reference_strategy,
    )


@app.command()
def tsnr(
    denoised_dir: ExistingDirectory,
    fmriprep_dir: ExistingDirectory,
    output_dir: Path,
    sub: list[str] | None = None,
    task_label: str | None = None,
    tr: float | None = None,
    high_pass: float = 0.01,
    n_jobs: int = 4,
    dataset: DatasetName = "ibc",
):
    """tSNR of each denoised run and of its fMRIPrep source, same definition for every dataset.

    `denoised_dir` mirrors `fmriprep_dir` (output of `denoise`). tSNR =
    mean / std over time, the std taken after removing a cosine drift basis
    at `high_pass` Hz (default: the LSS GLM's 0.01 Hz; 0 keeps the raw std).
    Voxels left at zero by denoising are excluded from both versions.
    """
    from tsnr import run_tsnr

    ds = DATASETS[dataset]
    run_tsnr(
        denoised_dir,
        fmriprep_dir,
        output_dir,
        task_label or ds.task_label,
        tr or ds.tr,
        sub_labels=sub,
        high_pass=high_pass or None,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    app()
