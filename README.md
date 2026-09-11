# IBC RSVP denoising vs. decoding

Does denoising fMRI data improve brain decoding, and does the order of
operations matter? This project compares three preprocessing strategies for
the IBC dataset's RSVP language task:

1. **fMRIPrep-only** — no denoising.
2. **fMRIPrep + Denoising** — denoise after fMRIPrep (in T1w space).
3. **Denoising + fMRIPrep** — denoise the raw BOLD, then run fMRIPrep on the
   denoised data.

Denoising uses [`patch-denoise`](https://github.com/paquiteau/patch-denoising)
(patch-based local low-rank denoising: MP-PCA, Optimal Thresholding, NORDIC,
...). Decoding is a per-trial LSS GLM (GPU-accelerated) feeding an ANOVA +
linear SVM classifier, cross-validated by leaving runs out.

## Dataset layout

```
/volatile2/ibc_rsvp/
├── sourcedata/                      BIDS raw data (IBC, task-RSVPLanguage)
│   └── sub-XX/ses-YY/{anat,fmap,func}/...
└── derivatives/
    ├── fmriprep/                    produced externally (not by this repo)
    ├── fmriprep+denoised/           `denoise` after fmriprep
    └── denoised+fmriprep/           `denoise` before fmriprep
```

Subjects: `01, 02, 04, 05, 06, 07, 08, 09, 11, 12, 13, 14, 15` (see
`_constants.SUBJECTS`). One RSVPLanguage session per subject, 6 runs/session,
TR=2.0s, 6-way condition decoding (`_constants.CHANCE = 1/6`).

## Setup

```bash
uv sync
```

Requires a CUDA-capable GPU (the LSS GLM regression is GPU-only by design) and
the `patch-denoise[gpu]` editable dependency at
`/home/pc280686/gits/nsp/patch-denoising` to exist on the machine.

## Running the three strategies

`derivatives_folder`/`mask_derivatives_folder`/`--output-dir` should be passed
as **absolute paths** — an absolute path overrides the dataset root in the
internal path join, which is the simplest way to point at a derivatives tree
outside the raw dataset directory.

### 1. fMRIPrep-only

```bash
uv run python cli.py glm /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    --desc preproc

uv run python cli.py decode /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/decode_results/fmriprep
```

### 2. fMRIPrep + Denoising

```bash
uv run python cli.py denoise /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/derivatives/fmriprep+denoised \
    --space-label T1w --mask-source fmriprep \
    --fmriprep-dir /volatile2/ibc_rsvp/derivatives/fmriprep

uv run python cli.py glm /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/fmriprep+denoised \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    --desc denoised

uv run python cli.py decode /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/fmriprep+denoised \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/decode_results/fmriprep+denoised
```

### 3. Denoising + fMRIPrep

```bash
uv run python cli.py denoise /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/denoised+fmriprep \
    --mask-source auto

# --- external step, outside this CLI's scope ---
# Run fMRIPrep yourself on derivatives/denoised+fmriprep as if it were a new
# rawdata folder, producing e.g. derivatives/denoised+fmriprep_fmriprepped/.
# ------------------------------------------------

uv run python cli.py glm /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/denoised+fmriprep_fmriprepped \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    --desc preproc

uv run python cli.py decode /volatile2/ibc_rsvp/sourcedata \
    /volatile2/ibc_rsvp/derivatives/denoised+fmriprep_fmriprepped \
    /volatile2/ibc_rsvp/derivatives/fmriprep \
    /volatile2/ibc_rsvp/decode_results/denoised+fmriprep
```

`mask_derivatives_folder` is always `derivatives/fmriprep` — brain masks are
anatomical and identical across strategies, so they're always sourced from
the plain fMRIPrep output, never from a denoised derivatives folder (which
may not contain a mask at all).

### 4. Compare the three strategies

`analysis` reads all three `decode --output-dir` trees back together, so give
them a shared parent and one subdirectory per strategy name:

```bash
uv run python cli.py decode ... /volatile2/ibc_rsvp/decode_results/fmriprep
uv run python cli.py decode ... /volatile2/ibc_rsvp/decode_results/fmriprep+denoised
uv run python cli.py decode ... /volatile2/ibc_rsvp/decode_results/denoised+fmriprep

uv run python cli.py analysis /volatile2/ibc_rsvp/decode_results \
    /volatile2/ibc_rsvp/analysis_results
```

Output:

```
analysis_results/
├── subjects/sub-XX/pairwise_ttests.csv, accuracy_boxplot.png, confusion_matrix_<strategy>.png
└── group/lmm_summary.txt, pairwise_contrasts.csv, subject_level_pairwise_ttests.csv, group_comparison.png
```


## Statistics methodology

- **Subject-level**: paired t-test with the Nadeau-Bengio (2003) correction
  for cross-validation fold correlation, applied to every pairwise strategy
  comparison per subject. `n_train`/`n_test` are derived from the actual CV
  fold sizes recorded by `decode` (`fold_sizes.json`) — not a hardcoded
  constant.
- **Group-level**: one linear mixed-effects model,
  `accuracy ~ strategy` with subject as a random effect, fitted once. All 3
  pairwise strategy contrasts are derived from that single fit (no repeated
  re-fitting, no extra multiple-comparison correction needed since these are
  contrasts of one model, not independent tests).

## Future work

- **Fused GLM+decode pipeline**: currently `glm` and `decode` are separate
  stages that round-trip through per-trial beta/z-score NIfTI files on disk.
  Since the GLM already runs on GPU, a fused pipeline that decodes directly
  from GPU tensors (skipping the disk round-trip) could be both faster and
  much lighter on disk space.
- **Denoising parameter sweeps**: `denoise`'s method/patch-shape/overlap are
  already full CLI flags — a sweep just needs repeated invocations.
- **No automated test suite** yet.
- The external fMRIPrep step for the Denoising+fMRIPrep strategy is manual
  and outside this CLI's scope.
- `decode`'s CV group key (`sub-run`) assumes one session per subject, true
  for all current IBC RSVP subjects — would need revisiting for a
  multi-session subject.

## Maintenance utilities (you probably don't need them)

`scripts/fix1run/fix_symlinks.py` repoints stale symlinks in `sourcedata/`/
`derivatives/` from an old storage path to the current one:

```bash
python scripts/fix1run/fix_symlinks.py [--dry-run] <root_dir> [<root_dir> ...]
```
