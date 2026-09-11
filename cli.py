"""Cyclopts CLI for generating the data and figures."""

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.types import ExistingDirectory
from patch_denoise.bindings.cli import parse_dims
from patch_denoise.bindings.utils import fast_cuda_check
from patch_denoise.space_time.base import DenoiserName

from _constants import (
    DEFAULT_SPACE_LABEL,
    DEFAULT_TR,
    DEFAULT_TASK_LABEL,
    STRATEGIES,
    SUBJECTS,
)

app = App()

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
    task_label: str = DEFAULT_TASK_LABEL,
    desc: str = "preproc",
    tr: float = DEFAULT_TR,
    space_label: str = DEFAULT_SPACE_LABEL,
    sub: list[str] | None = None,
    save_betas: Annotated[bool, Parameter(negative="--no-betas")] = True,
    n_jobs: int = -1,
):
    """Compute per-trial LSS GLM beta/z-score maps for all subjects and sessions.

    `derivatives_folder`/`mask_derivatives_folder` should be absolute paths
    (e.g. /volatile2/ibc_rsvp/derivatives/fmriprep+denoised) — an absolute path
    here overrides `dataset_dir` in the internal join, which is the simplest way
    to point at a derivatives tree outside the raw dataset directory.
    """
    from glm import make_lss_betas, rsvp_cleanup_events_df

    make_lss_betas(
        dataset_dir,
        task_label,
        derivatives_folder,
        desc,
        tr,
        mask_derivatives_folder,
        sub_labels=sub,
        space_label=space_label,
        cleanup_func=rsvp_cleanup_events_df,
        save_betas=save_betas,
        n_jobs=n_jobs,
    )


@app.command()
def denoise(
    input_dir: ExistingDirectory,
    output_dir: Path,
    task_label: str = DEFAULT_TASK_LABEL,
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
        task_label,
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


@app.command()
def decode(
    dataset_dir: ExistingDirectory,
    derivatives_folder: Path,
    mask_derivatives_folder: Path,
    output_dir: Path,
    sub: list[str] | None = None,
    task_label: str = DEFAULT_TASK_LABEL,
    glm_folder: str = "glm",
    suffix: str = "zscore",
    n_jobs: int = 1,
    k: int = 1500,
    nested: bool = False,
    leave_p_out: list[int] | None = None,
    tag: str = "decode",
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
    """
    from decode import (
        compare_group_sizes,
        load_glm_data,
        run_decoding,
        save_group_size_comparison,
        save_results,
    )

    for sub_label in sub or SUBJECTS:
        X, y, g = load_glm_data(
            dataset_dir,
            derivatives_folder,
            mask_derivatives_folder,
            task_label=task_label,
            glm_folder=glm_folder,
            suffix=suffix,
            sub_labels=[sub_label],
            n_jobs=n_jobs,
        )
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
        else:
            results, predictions = compare_group_sizes(
                X,
                y,
                g,
                group_sizes=tuple(leave_p_out),
                k=k,
                n_jobs=n_jobs,
            )
            save_group_size_comparison(
                output_dir, sub_label, results, k, predictions, y, tag
            )


@app.command()
def analysis(
    decode_dir: ExistingDirectory,
    output_dir: Path,
    strategy: list[str] | None = None,
    sub: list[str] | None = None,
    tag: str = "decode",
    k: int = 1500,
    reference_strategy: str = "fmriprep",
):
    """Subject-level (paired Nadeau-Bengio t-test) and group-level (LMM) stats + figures.

    `decode_dir` must contain one subdirectory per strategy, each populated by
    a `decode --output-dir <decode_dir>/<strategy>` run (see README).
    """
    from analysis import run_analysis

    run_analysis(
        decode_dir,
        output_dir,
        strategies=strategy or list(STRATEGIES),
        subjects=sub or SUBJECTS,
        tag=tag,
        k=k,
        reference_strategy=reference_strategy,
    )


if __name__ == "__main__":
    app()
