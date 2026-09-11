"""Compute per-trial LSS GLMs and save the resulting beta and z-score maps to disk."""

import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from typing import Callable, cast
import numpy as np
import torch
from joblib import Parallel, delayed
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import NiftiMasker
from tqdm.auto import tqdm


def snake_to_camel(text):
    return "".join(word.capitalize() for word in text.split("_"))


def find_brain_mask(dataset_path, sub_label, fmriprep_folder, space_label="T1w"):
    """Locate this subject's fMRIPrep brain mask.

    Brain masks are anatomical (derived from the T1w segmentation) and
    identical regardless of BOLD denoising, so they are always sourced from
    `fmriprep_folder` (the plain fMRIPrep derivatives), never from whatever
    `derivatives_folder` is being GLM'd — which may be a denoised variant
    containing no mask files at all.
    """
    mask_root = Path(dataset_path) / fmriprep_folder
    candidates = sorted(
        mask_root.glob(
            f"sub-{sub_label}/ses-*/func/*_space-{space_label}_desc-brain_mask.nii.gz"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No space-{space_label} brain mask found for sub-{sub_label} under {mask_root}"
        )
    return candidates[0]


def find_t1w(dataset_path, sub_label, fmriprep_folder):
    """Locate this subject's fMRIPrep-preprocessed native-space T1w image.

    Used as the anatomical background for plotting — native T1w space, same as
    `find_brain_mask`'s `space-T1w` functional data, so no resampling is needed.
    """
    anat_root = Path(dataset_path) / fmriprep_folder
    candidates = sorted(
        anat_root.glob(f"sub-{sub_label}/ses-*/anat/*_desc-preproc_T1w.nii.gz")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No desc-preproc T1w image found for sub-{sub_label} under {anat_root}"
        )
    return candidates[0]


def load_confounds_manual(run_file, motion_columns=None, demean=True):
    """
    Load basic motion confounds directly from the fMRIPrep-style confounds TSV,
    bypassing nilearn.interfaces.fmriprep.load_confounds().

    nilearn's load_confounds() only recognizes a small whitelist of fMRIPrep-native
    'desc' tags (desc-preproc, desc-smoothAROMAnonaggr, etc.) to validate the BOLD
    filename and locate the matching confounds file. Custom tags like
    'desc-denoised' (produced by our own denoising pipeline) aren't recognized,
    causing a cryptic "Invalid file type" error even though the file is valid.

    Since we already know the exact confounds filename convention
    (<bold_stem>_desc-confounds_timeseries.tsv, sitting next to the bold file),
    we load it directly instead.
    """
    if motion_columns is None:
        motion_columns = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    run_path = Path(run_file)
    stem = run_path.name

    # Strip "_space-*_desc-*_bold.nii.gz" (or plain "_bold.nii.gz") to recover
    # the shared BIDS stem used by the confounds file.
    if "_space-" in stem:
        stem = stem.split("_space-")[0]
    else:
        stem = stem.replace("_bold.nii.gz", "").replace("_bold.nii", "")

    confounds_path = run_path.parent / f"{stem}_desc-confounds_timeseries.tsv"

    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")

    confounds_df = pd.read_csv(confounds_path, sep="\t")

    missing = [c for c in motion_columns if c not in confounds_df.columns]
    if missing:
        raise KeyError(f"Missing expected motion columns {missing} in {confounds_path}")

    filtered_confounds = confounds_df[motion_columns].fillna(0).reset_index(drop=True)

    if demean:
        filtered_confounds = filtered_confounds - filtered_confounds.mean()

    sample_mask = None  # no scrubbing requested (matches strategy=["motion"] behavior)
    return filtered_confounds, sample_mask


def rsvp_cleanup_events_df(events_df):
    """Sanitize the task name for RSVP, pass as `cleanup_function` to make_lss_betas."""
    events_df["trial_type"] = events_df["trial_type"].replace(
        {
            "simple_sentence.*": "simple_sentence",
            "complex_sentence.*": "complex_sentence",
        },
        regex=True,
    )
    return events_df


def lss_transformer(events_df, row_number):
    """Label one trial for one LSS model."""
    trial_condition = events_df.loc[row_number, "trial_type"]
    trial_type_series = events_df["trial_type"].loc[
        events_df["trial_type"] == trial_condition
    ]
    trial_number = trial_type_series.index.tolist().index(row_number)
    trial_name = f"{trial_condition}__{trial_number:03d}"
    events_df.loc[row_number, "trial_type"] = trial_name
    return events_df, trial_name


def _make_dm_lss_batch(
    events_df, frame_times, run_confound, ignore_trial="probe", **design_params
):
    """Make a batch of LSS design matrices."""
    dm_stack, trial_names, target_indices = [], [], []
    events_df_filter = events_df.loc[
        events_df["trial_type"] != ignore_trial
    ]  # only compute events for non-probe events
    for i_trial in events_df_filter.index:
        lss_events_df, trial_name = lss_transformer(events_df.copy(), i_trial)
        lss_events_df = lss_events_df[lss_events_df["trial_type"] != ignore_trial]
        dm = make_first_level_design_matrix(
            frame_times=frame_times,
            events=lss_events_df,
            add_regs=run_confound,
            **design_params,
        )
        dm_stack.append(dm.values)
        target_indices.append(list(dm.columns).index(trial_name))
        trial_names.append(trial_name)
    return dm_stack, trial_names, target_indices


@torch.no_grad()
def _compute_zscore_batch_gpu(
    dm_stack, fmri_data_clean, target_indices, VOXEL_CHUNK=20_000
):
    """Compute z-scores for a batch of trials on the GPU."""
    gpu_Y = torch.from_numpy(fmri_data_clean.astype(np.float32)).cuda()
    gpu_X = torch.from_numpy(np.array(dm_stack).astype(np.float32)).cuda()
    gpu_target_idx = torch.tensor(target_indices).cuda()

    voxel_mean = torch.mean(gpu_Y, dim=0)
    safe_mean = torch.clamp(voxel_mean, min=1.0)
    gpu_Y = 100.0 * (gpu_Y / safe_mean - 1.0)

    gpu_pinv = torch.linalg.pinv(gpu_X)
    gpu_xtx_inv = torch.linalg.pinv(torch.bmm(gpu_X.transpose(1, 2), gpu_X))

    batch_idx = torch.arange(len(dm_stack)).cuda()
    xtx_inv_target = gpu_xtx_inv[batch_idx, gpu_target_idx, gpu_target_idx].unsqueeze(1)

    n_time, n_voxels = gpu_Y.shape
    n_preds = gpu_X.shape[2]
    dof = n_time - n_preds

    all_z_scores, all_betas = [], []
    for v in range(0, n_voxels, VOXEL_CHUNK):
        v_end = min(v + VOXEL_CHUNK, n_voxels)
        Y_chunk = gpu_Y[:, v:v_end]
        beta_chunk = torch.matmul(gpu_pinv, Y_chunk)
        Y_pred_chunk = torch.matmul(gpu_X, beta_chunk)
        res_chunk = Y_pred_chunk - Y_chunk.unsqueeze(0)
        sigma_sq = torch.sum(res_chunk**2, dim=1) / dof
        se_betas = torch.sqrt(sigma_sq * xtx_inv_target)
        target_betas = beta_chunk[batch_idx, gpu_target_idx, :]
        all_z_scores.append((target_betas / se_betas).cpu())
        all_betas.append(target_betas.cpu())

    return torch.cat(all_z_scores, dim=1).numpy(), torch.cat(all_betas, dim=1).numpy()


def _load_run(run_file, events_df, cleanup_func, masker):
    """Read+mask one run's BOLD data and load its confounds — the I/O-bound half of a run.

    Independent across runs (each reads its own file into its own array), so safe
    to run concurrently across runs of a subject on one shared, already-fitted masker.
    """
    if cleanup_func is not None:
        events_df = cleanup_func(events_df)
    events_df = events_df.reset_index(drop=True)
    filtered_confounds, _sample_mask = load_confounds_manual(run_file)
    fmri_data_clean = masker.transform(run_file)
    return fmri_data_clean, events_df, filtered_confounds


def _write_run_outputs(glm_dir, keys, betas_org, zscores_org, masker, save_betas):
    """Write one run's per-condition beta/z-score maps to disk — the I/O-bound tail of a run."""
    if save_betas:
        keys["suffix"] = "effectsize"
        for task_cond, b in betas_org.items():
            full_map = masker.inverse_transform(np.stack(b))
            keys["entities"]["cond"] = snake_to_camel(task_cond)
            full_map.to_filename(
                glm_dir
                / create_bids_filename(
                    keys, entities_to_include=bids_entities()["raw"] + ["desc", "cond"]
                )
            )

    keys["suffix"] = "zscore"
    for task_cond, z in zscores_org.items():
        full_map = masker.inverse_transform(np.stack(z))
        keys["entities"]["cond"] = snake_to_camel(task_cond)
        full_map.to_filename(
            glm_dir
            / create_bids_filename(
                keys, entities_to_include=bids_entities()["raw"] + ["desc", "cond"]
            )
        )


def make_lss_betas(
    dataset_path,
    task_label,
    derivatives_folder,
    desc,
    t_r,
    mask_derivatives_folder,
    sub_labels=None,
    space_label="T1w",
    cleanup_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    save_betas: bool = True,
    n_jobs: int = -1,
    **kwargs,
):
    """Make LSS GLMs for each trial in the dataset and save the resulting beta and z-score maps to disk.

    Parameters
    ----------
    dataset_path : str
        Path to the BIDS dataset.
    task_label : str
        Task label to process.
    derivatives_folder : str
        Name of the derivatives folder where the GLM results will be saved.
    desc : str
        Description of the GLM, used in the output filenames.
    t_r : float
        Repetition time of the fMRI data.
    mask_derivatives_folder : str
        Derivatives folder holding the plain fMRIPrep output, used to locate each
        subject's brain mask (see `find_brain_mask`) regardless of which
        `derivatives_folder` is actually being GLM'd.
    sub_labels : list of str, optional
        List of subject labels to process. If None, all subjects will be processed.
    space_label : str, optional
        Space label to use for the output files. Default is "T1w".
    cleanup_func : callable, optional
        Function to clean up the events DataFrame before processing. If None, no cleanup will be
        performed. The function should take a DataFrame as input and return a cleaned DataFrame.
    save_betas : bool, optional
        Whether to write per-trial effectsize (beta) maps to disk, in addition to the
        z-score maps. Z-scores are always written; disable this to save disk space
        when only z-score maps are needed downstream (e.g. for decoding).
    n_jobs : int, optional
        Number of threads used to overlap each run's I/O (reading+masking the BOLD
        file, and writing the resulting beta/z-score maps) across runs of a subject.
        The GPU regression itself stays single-threaded/serial (one shared GPU) —
        this only parallelizes the disk read/write on either side of it, which
        profiling showed dominates wall time (~85% of it, vs ~10% GPU compute).
        -1 (default) uses all cores; a thread pool is used since gzip
        decompression/compression and disk I/O release the GIL, and passing the
        large per-run arrays between processes would itself be expensive.

    """

    models, *run_event_confoundss = first_level_from_bids(
        dataset_path=dataset_path,
        task_label=task_label,
        sub_labels=sub_labels,
        n_jobs=2,
        derivatives_folder=str(derivatives_folder),
        space_label=space_label,
        noise_model="ols",
        verbose=1,
        t_r=t_r,
        img_filters=[("desc", desc)],
        **kwargs,
    )

    glm_parameters = models[0].get_params()
    design_params = {}
    for k in [
        "drift_model",
        "drift_order",
        "fir_delays",
        "high_pass",
        "hrf_model",
        "min_onset",
    ]:
        design_params[k] = glm_parameters.pop(k)

    # first_level_from_bids's return type is a broad union covering unrelated call
    # signatures; with our arguments this element is always the per-run events list.
    events_lists = cast(list[list], run_event_confoundss[0])
    progress_bar = tqdm(total=sum(len(a) for a in events_lists))

    for run_events_confounds in zip(*run_event_confoundss):
        first_run_file = run_events_confounds[0][0]
        this_sub_label = parse_bids_filename(first_run_file)["entities"]["sub"]
        mask_path = find_brain_mask(
            dataset_path,
            this_sub_label,
            fmriprep_folder=mask_derivatives_folder,
            space_label=space_label,
        )
        masker = NiftiMasker(mask_img=mask_path).fit()
        runs = list(zip(*run_events_confounds))

        # Read+mask all of this subject's runs concurrently: profiling showed this
        # (plus the write phase below) is ~85% of wall time per run, vs ~10% GPU
        # compute — threaded since gzip decompression/nibabel I/O release the GIL.
        loaded_runs = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_load_run)(run_file, events_df, cleanup_func, masker)
            for run_file, events_df, _run_confound in runs
        )

        write_jobs = []
        for (run_file, _events_df, _run_confound), (
            fmri_data_clean,
            events_df,
            run_confound,
        ) in zip(runs, loaded_runs):
            frame_times = np.arange(fmri_data_clean.shape[0]) * glm_parameters["t_r"]

            # create all the design matrices for the LSS models, and get the corresponding trial names and target indices
            dm_stack, trial_name, target_indices = _make_dm_lss_batch(
                events_df, frame_times, run_confound, **design_params
            )
            # GPU goes brrrrr.... (kept serial: one shared GPU across all runs)
            z_scores, betas = _compute_zscore_batch_gpu(
                dm_stack, fmri_data_clean, target_indices
            )

            keys = parse_bids_filename(run_file)
            entities = keys["entities"]
            glm_dir = (
                Path(dataset_path)
                / derivatives_folder
                / f"sub-{entities['sub']}"
                / f"ses-{entities['ses']}"
                / "glm"
            )
            os.makedirs(glm_dir, exist_ok=True)

            betas_org, zscores_org = defaultdict(list), defaultdict(list)
            for t, z, b in zip(trial_name, z_scores, betas):
                zscores_org[t.split("__")[0]].append(z)
                betas_org[t.split("__")[0]].append(b)

            write_jobs.append((glm_dir, keys, betas_org, zscores_org))
            progress_bar.update()

        # Write all of this subject's run outputs concurrently, same reasoning as the read phase.
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_write_run_outputs)(
                glm_dir, keys, betas_org, zscores_org, masker, save_betas
            )
            for glm_dir, keys, betas_org, zscores_org in write_jobs
        )
    progress_bar.close()
