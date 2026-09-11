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
├── subjects/sub-XX/pairwise_ttests.csv, accuracy_boxplot.png,
│                    confusion_matrix_<strategy>.png, weight_map_<condition>.png
└── group/lmm_summary.txt, pairwise_contrasts.csv, subject_level_pairwise_ttests.csv, group_comparison.png
```

`weight_map_<condition>.png` compares, per subject and condition, the SVM's
weight map (averaged over CV folds) across denoising strategies, overlaid on
the subject's T1w with transparency thresholding. The underlying weights are
written by `decode` as a single 4D `weight_maps.nii.gz` (one volume per
condition, order given by the `weight_maps_labels.json` sidecar) alongside its
other per-subject outputs.

### Caching repeated GLM-data loads

`decode`'s `--cache-dir <dir>` caches each subject's loaded beta/z-score
arrays, keyed by derivatives folder + subject (independent of `k`,
`--leave-p-out`, `--tag`) — useful when sweeping several `k` values for the
same subject/strategy, since it skips re-parsing NIfTI files on a cache hit.
Safe to share across concurrent `decode` invocations for different
subjects/k values (e.g. a parallel job runner) — see `scripts/grid_search.sh`
below. `X` is loaded as float32 (the source NIfTI's native precision; nilearn
upcasts to float64 by default, adding no real precision but doubling memory
and BLAS cost) — re-running an old decode after upgrading to this may shift
`cvscores.npy` at floating-point-noise level versus a result computed before.

### `k`/`p` grid search

`grid-analysis` aggregates decode runs across several `k` (ANOVA feature
count) and `p` (`--leave-p-out`, i.e. training-set size) values into one
comparison — a different, more exploratory question than `analysis`'s
"is strategy A better at one fixed k":

```bash
uv run python cli.py grid-analysis /volatile2/ibc_rsvp/decode_results \
    /volatile2/ibc_rsvp/analysis_results_grid \
    --strategy fmriprep --strategy fmriprep+denoised \
    --k 1500 --k 3000 --k 10000 --k 20000 --k 50000 \
    --p 1 --p 2 --p 3 --p 4 --p 5
```

It expects, per strategy and per `k`, both a `--tag decode` tree (p=1,
read from there) and a `--tag cv_sweep` tree (`--leave-p-out 2 3 4 5` —
never re-run p=1 there, it would just duplicate the main decode). Output:
`grid_long.csv`/`grid_summary.csv` (mean/std/sem per (strategy, k, p)),
`heatmap_accuracy.png` (accuracy over (k, p), one panel per strategy plus a
`gap = other - reference` panel), `lines_accuracy_vs_k.png` (accuracy vs k,
one line per p).

`scripts/grid_search.sh` runs the many `decode` invocations a full grid
needs as separate, bounded-concurrency processes (tunable via
`MAX_CONCURRENT_JOBS`/`NJOBS_PER_JOB` env vars — budget their product
against physical cores, not hyperthreads), skips any `(strategy, k, subject)`
job whose output already exists, and calls `grid-analysis` once all jobs
finish. Edit the `STRATEGIES`/`MAIN_KS`/`SWEEP_KS` arrays at the top to
extend a sweep — already-done combinations skip instantly.

## Statistics methodology

- **Subject-level**: paired t-test with the Nadeau-Bengio (2003) correction
  for cross-validation fold correlation, applied to every pairwise strategy
  comparison per subject. `n_train`/`n_test` are derived from the actual CV
  fold sizes recorded by `decode` (`fold_sizes.json`) — not a hardcoded
  constant.
- **Group-level**: one linear mixed-effects model, `accuracy ~ strategy`,
  fitted once on the fold-level data (not subject means) with subject as a
  random effect (intercept). A random slope on strategy was tried too, but
  doesn't converge for this dataset's subject count — the slope variance
  estimates to the parameter-space boundary — so the random-intercept model
  is used. All 3 pairwise strategy contrasts are derived from that single fit
  (no repeated re-fitting, no extra multiple-comparison correction needed
  since these are contrasts of one model, not independent tests).
- **Group-level, nonparametric check**: a hierarchical (subject-then-fold)
  bootstrap on the same fold-level data — resample subjects with
  replacement, then resample folds within each resampled subject with
  replacement, recomputing the group contrast each replicate. Needs no REML
  convergence, so it's a useful cross-check on the LMM p-values, especially
  given the boundary-solution warning above. Written to
  `group/bootstrap_pairwise_contrasts.csv` (`estimate`/`ci_low`/`ci_high`/
  `p_value` per pair, sign convention `strategy_b - strategy_a`).

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
