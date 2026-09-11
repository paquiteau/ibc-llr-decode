"""Run patch-denoise over a BIDS(-like) tree of BOLD runs."""

import shutil
from pathlib import Path

from nilearn.interfaces.bids import parse_bids_filename
from nilearn.interfaces.bids.utils import bids_entities, create_bids_filename
from nilearn.maskers import NiftiMasker
from patch_denoise.bindings.utils import DENOISER_MAP, load_as_array, save_array
from patch_denoise.space_time.base import _patch_param
from tqdm.auto import tqdm


def find_bold_files(input_root, task_label, sub_labels=None, space_label=None):
    """Walk sub-*/ses-*/func/*_bold.nii.gz under input_root, BIDS-style.

    `space_label=None` matches raw (not spatially-normalized) BOLD files, i.e.
    files with no `space` entity at all — appropriate for denoising raw
    `sourcedata`. Pass e.g. `space_label="T1w"` to target fMRIPrep output instead.
    """
    root = Path(input_root)
    subs = sub_labels or [
        p.name.removeprefix("sub-") for p in sorted(root.glob("sub-*")) if p.is_dir()
    ]
    files = []
    for sub in subs:
        for f in sorted(
            root.glob(f"sub-{sub}/ses-*/func/*_task-{task_label}*_bold.nii.gz")
        ):
            entities = parse_bids_filename(f)["entities"]
            has_space = "space" in entities
            if space_label is not None and entities.get("space") != space_label:
                continue
            if space_label is None and has_space:
                continue
            files.append(f)
    return files


def denoised_output_path(bold_file, output_root):
    """Mirror bold_file's sub/ses/func path under output_root, with desc set to 'denoised'."""
    keys = parse_bids_filename(bold_file)
    keys["entities"]["desc"] = "denoised"
    entities = keys["entities"]
    out_name = create_bids_filename(
        keys, entities_to_include=bids_entities()["raw"] + ["space", "desc"]
    )
    return (
        Path(output_root)
        / f"sub-{entities['sub']}"
        / f"ses-{entities['ses']}"
        / "func"
        / out_name
    )


def _copy_sidecars(bold_file, output_file):
    """Copy the sidecars a downstream GLM step needs next to the denoised bold file.

    Confounds files are copied under their ORIGINAL name (glm.load_confounds_manual
    derives the expected confounds filename purely from the bold file's own stem),
    not renamed to match the new desc-denoised bold stem. Silently skips any
    sidecar that doesn't exist (e.g. sourcedata has no confounds file before
    fMRIPrep has run).
    """
    bold_file = Path(bold_file)
    output_file = Path(output_file)
    stem = (
        bold_file.name.split("_space-")[0]
        if "_space-" in bold_file.name
        else (bold_file.name.replace("_bold.nii.gz", "").replace("_bold.nii", ""))
    )
    candidates = [
        bold_file.with_name(
            bold_file.name.replace(".nii.gz", ".json").replace(".nii", ".json")
        ),
        bold_file.parent / f"{stem}_desc-confounds_timeseries.tsv",
        bold_file.parent / f"{stem}_desc-confounds_timeseries.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, output_file.parent / candidate.name)


def denoise_one_run(
    bold_file,
    output_file,
    *,
    method="optimal-fro",
    patch_shape=(11, 11, 11, -1),
    patch_overlap=(5, 5, 5, -1),
    recombination="weighted",
    mask_threshold=50,
    mask_file=None,
    gpu=True,
):
    """Denoise a single 4D BOLD run and write the result + sidecars next to it."""
    input_data, affine = load_as_array(Path(bold_file))

    if mask_file is not None:
        mask_data, _ = load_as_array(Path(mask_file))
        mask_data = mask_data.astype(bool)
    else:
        # No pre-existing compatible mask (e.g. denoising raw sourcedata before
        # fMRIPrep has run): auto-compute an EPI mask, like patch-denoise's own
        # CLI does by default.
        masker = NiftiMasker(mask_strategy="epi").fit(str(bold_file))
        mask_data = masker.mask_img_.get_fdata().astype(bool)

    patch_shape_ = _patch_param(patch_shape, input_data.shape)
    patch_overlap_ = _patch_param(patch_overlap, input_data.shape)

    if gpu:
        from patch_denoise.gpu.main import GPU_SUPPORTED_METHODS, main_gpu

        if method not in GPU_SUPPORTED_METHODS:
            raise ValueError(
                f"Method {method!r} has no GPU implementation "
                f"(supported: {sorted(GPU_SUPPORTED_METHODS)}). Pass gpu=False or change method."
            )
        result = main_gpu(
            input_data,
            patch_shape=patch_shape_,
            patch_overlap=patch_overlap_,
            mask_threshold=mask_threshold,
            recombination=recombination,
            method=method,
            mask=mask_data,
        )
    else:
        denoise_func = DENOISER_MAP[method]
        result = denoise_func(
            input_data,
            patch_shape=patch_shape_,
            patch_overlap=patch_overlap_,
            mask=mask_data,
            mask_threshold=mask_threshold,
            recombination=recombination,
        )
    denoised_data = result[0]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_array(denoised_data, affine, output_file)
    _copy_sidecars(bold_file, output_file)
    return output_file


def run_denoising(
    input_root,
    output_root,
    task_label,
    *,
    sub_labels=None,
    space_label=None,
    method="optimal-fro",
    patch_shape=(11, 11, 11, -1),
    patch_overlap=(5, 5, 5, -1),
    recombination="weighted",
    mask_threshold=50,
    mask_source="auto",
    fmriprep_root=None,
    gpu=True,
    overwrite=False,
):
    """Denoise every matching BOLD run under input_root, writing a mirrored derivatives tree.

    `mask_source="fmriprep"` reuses the fMRIPrep brain mask (via `glm.find_brain_mask`)
    and is only correct when denoising already-fMRIPrepped, space-normalized data
    (i.e. `space_label` set to the fMRIPrep space, e.g. "T1w") — using a T1w-space
    mask on raw, un-normalized sourcedata would be a space/resolution mismatch.
    `mask_source="auto"` computes a per-run EPI mask instead, and is the only
    valid choice when denoising raw sourcedata.
    """
    from glm import find_brain_mask  # single source of truth for anatomical masks

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    desc_json = output_root / "dataset_description.json"
    if not desc_json.exists():
        desc_json.write_text(
            '{"Name": "patch-denoise output", "BIDSVersion": "1.8.0", '
            '"DatasetType": "derivative"}'
        )

    if mask_source == "fmriprep" and fmriprep_root is None:
        raise ValueError("mask_source='fmriprep' requires fmriprep_root to be set")
    if mask_source not in ("auto", "fmriprep"):
        raise ValueError(
            f"mask_source must be 'auto' or 'fmriprep', got {mask_source!r}"
        )

    bold_files = find_bold_files(
        input_root, task_label, sub_labels=sub_labels, space_label=space_label
    )
    written = []
    for bold_file in tqdm(bold_files, desc="Denoising runs"):
        output_file = denoised_output_path(bold_file, output_root)
        if output_file.exists() and not overwrite:
            continue
        mask_file = None
        if mask_source == "fmriprep":
            sub_label = parse_bids_filename(bold_file)["entities"]["sub"]
            mask_file = find_brain_mask(
                input_root,
                sub_label,
                fmriprep_folder=fmriprep_root,
                space_label=space_label or "T1w",
            )
        written.append(
            denoise_one_run(
                bold_file,
                output_file,
                method=method,
                patch_shape=patch_shape,
                patch_overlap=patch_overlap,
                recombination=recombination,
                mask_threshold=mask_threshold,
                mask_file=mask_file,
                gpu=gpu,
            )
        )
    return written
