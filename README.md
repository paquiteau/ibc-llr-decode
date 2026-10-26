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
`_constants.DATASETS["ibc"]`). One RSVPLanguage session per subject, 6
runs/session, TR=2.0s, 6-way condition decoding (chance 1/6).

### Second dataset: 7T stroke patients (`--dataset motif`)

Three patients (`p01, p02, p03`), task `motif4limbs`, 3 runs of 96 trials
(mini-blocks of 4 trials of the same limb), TR=1.6s, no session level,
4-way limb decoding (chance 1/4). `glm`, `denoise`, `decode`, `analysis` and
`tsnr` take `--dataset motif` for these defaults (subjects, task, TR,
training sizes, mini-block size of the single-subject test). The source
data (`/neurospin/motif-stroke/7T_protocol/patients`) has no `events.tsv`;
`scripts/motif_stroke/prepare_bids.py` builds them from the behavioural logs
and lays out `/volatile2/decoding-motif-stroke/`:

```
/volatile2/decoding-motif-stroke/
├── sourcedata/              raw BOLD (symlinks) + events.tsv from the logs
├── derivatives/fmriprep/    symlinks to the shared fMRIPrep outputs, so
│                            `glm` writes here, never into the shared tree
└── rois/                    sensorimotor ROI per patient (native T1w), for
                             `decode --roi-mask`
```

`scripts/homogenized_pipeline.sh` runs both datasets through the same GLM,
decoding, analysis and tSNR.

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

Every strategy is compared to `--reference-strategy` (default `fmriprep`). If
the `--tag cv_sweep --leave-p-out 2 3 4` decode outputs exist, the same run
also covers the training-size analysis (`--p`, default `1 2 3 4`). Output:

```
analysis_results/
├── group/
│   ├── summary.txt                     every number below, in words
│   ├── subject_accuracy.csv            subject x strategy accuracy
│   ├── group_contrasts.csv             main test (one row per comparison)
│   ├── subject_tests.csv               single-subject tests
│   ├── per_class_recall.csv            per-condition recall differences
│   ├── training_size_contrasts.csv     main test per number of training runs
│   ├── training_size_trend.csv         does the gain depend on training size
│   └── fig_accuracy, fig_training_size, fig_confusion (.pdf + .png)
└── subjects/sub-XX/tests.csv, accuracy_folds.pdf, confusion_matrices.pdf,
                    weight_map_<condition>.pdf, {weight,pattern}_map_norm.pdf
```

Figures are sized for the ISBI two-column template (`fig_confusion` spans
the text width, the others one column) with TrueType-embedded fonts; display
names for strategies and conditions are set in `_constants.py`.

`weight_map_<condition>.pdf` compares, per subject and condition, the SVM's
weight map (averaged over CV folds) across denoising strategies, overlaid on
the subject's T1w with transparency thresholding. The underlying weights are
written by `decode` as a single 4D `weight_maps.nii.gz` (one volume per
condition, order given by the `weight_maps_labels.json` sidecar) alongside its
other per-subject outputs.

### Region of interest

`decode --roi-mask <path>` decodes within brain mask ∩ ROI (`{sub}` in the
path is replaced by the subject label; the ROI is resampled to the mask's
grid if needed). ANOVA still selects `k` voxels, inside the ROI. Write ROI
runs to their own `--output-dir`; they share the cache with whole-brain runs.

### GLM output folder

`glm --glm-folder <name>` (default `glm`) names the per-subject output
folder; pass the same `--glm-folder` to `decode`. The GLM code computed
before the fix below writes to `glm/`; results from the fixed code use
`glm_v2/`, so both stay on disk. The fix: the LSS standard errors took
`pinv(X^T X)` in float32, which squares the design's condition number past
the pinv cutoff and inflated z-scores (up to ~15% for some IBC trials, ~200x
for zero-duration events, whose regressors are ~1e-3); columns are now
normalized and diag((X^T X)^-1) is taken from `pinv(X)` (matches a float64
reference to ~1e-5).

### tSNR

`tsnr <denoised dir> <fmriprep dir> <output dir>` computes, per run, the
median tSNR before/after denoising and the median voxelwise ratio, over the
fMRIPrep brain mask minus voxels left at zero by denoising. tSNR = mean / std
over time, the std taken after removing the LSS GLM's cosine drifts
(`--high-pass 0.01` Hz; `0` keeps the raw std).

### Caching repeated GLM-data loads

`decode`'s `--cache-dir <dir>` caches each subject's loaded beta/z-score
arrays, keyed by derivatives folder (+ `--glm-folder` when not the default)
+ subject (independent of `k`,
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

The design is fully paired and balanced: all strategies are decoded on the
same trials with the same CV folds, and every subject has the same runs and
trials (IBC: 6 x 60; stroke: 3 x 96). All tests therefore compare each strategy to the reference on
paired per-subject values (summary-statistic approach, Holmes & Friston
1998). For a balanced design this is the random-effects inference a mixed
model with a random subject slope on strategy would give, without its
convergence problems at 13 subjects. It never treats CV folds as independent
observations, so it stays valid when folds overlap (`LeavePGroupsOut`, p > 1).
Fold-level models are not used for that reason: a random-intercept model
there pseudo-replicates the strategy effect and gives anti-conservative
p-values, badly so when folds overlap.

- **Group level** (`group_contrasts.csv`): per-subject accuracy difference
  (mean over folds = pooled trial accuracy, test folds being equal-sized),
  tested with an exact sign-flip permutation test (all 2^13 sign patterns,
  primary p-value). Also reported: the paired t-test (same hypothesis,
  parametric) and its 95% CI, Cohen's d_z, the number of subjects improved,
  and the relative gain in above-chance accuracy. Holm across comparisons
  when several strategies are compared to the reference.
- **Training size** (`training_size_*.csv`): the same test at each number of
  held-out runs p (Holm across p), plus a test on per-subject OLS slopes of
  the gain against the number of training runs, i.e. whether denoising helps
  more when training data is scarce.
- **Single subject** (`subject_tests.csv`): exact sign-flip test on paired
  unit differences, the unit being a mini-block of consecutive same-condition
  trials (`Dataset.block_size`: 1 for IBC, 4 for the stroke task, whose 4
  trials 3.2 s apart are not independent). A unit's difference is its number
  of correct out-of-fold predictions with the strategy minus with the
  reference, summed over its trials and over every fold that tests them, so
  overlapping folds (p > 1, read from `cv_predict_folds_p<p>.npy`) are never
  treated as independent. With 1-trial units and p=1 it is the exact McNemar
  test. BH-FDR across subjects, per comparison and p. With 3 patients, the
  stroke analysis rests on these tests; its group test cannot go below
  p = 0.25.
- **Per class** (`per_class_recall.csv`): the group-level test on per-class
  recall differences (confusion-matrix diagonal), Holm across classes.

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
  for all current IBC RSVP and stroke subjects — would need revisiting for a
  multi-session subject.
- Voxels at the brain-mask boundary that patch-denoise leaves at zero (~1%)
  are not refilled with the fMRIPrep signal.

## Maintenance utilities (you probably don't need them)

`scripts/fix1run/fix_symlinks.py` repoints stale symlinks in `sourcedata/`/
`derivatives/` from an old storage path to the current one:

```bash
python scripts/fix1run/fix_symlinks.py [--dry-run] <root_dir> [<root_dir> ...]
```
