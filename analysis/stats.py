"""Subject-level and group-level statistics for comparing denoising strategies."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from nilearn.image import index_img, load_img
from scipy import stats


def nadeau_bengio_corrected_ttest(scores_a, scores_b, n_train, n_test):
    """Paired t-test with the Nadeau & Bengio (2003) correction for CV fold correlation.

    `n_train`/`n_test` are the actual number of trials in the train/test split
    of each fold (a scalar, or averaged if it varies slightly across folds).
    These must come from the real CV split sizes (see `decode.run_decoding`'s
    `fold_sizes` output) — never hardcode them, the correction is only valid
    for the CV scheme that actually produced `scores_a`/`scores_b`.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    j = len(diffs)
    if j < 2:
        raise ValueError("Need at least 2 CV folds for the corrected t-test.")
    n_train, n_test = float(np.mean(n_train)), float(np.mean(n_test))
    mean_diff, var_diff = diffs.mean(), diffs.var(ddof=1)
    if var_diff == 0:
        return (np.inf, 0.0) if mean_diff != 0 else (0.0, 1.0)
    correction = 1.0 / j + n_test / n_train
    t_stat = mean_diff / np.sqrt(correction * var_diff)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=j - 1)
    return float(t_stat), float(p_value)


def _condition_dir(decode_root, strategy, sub_label, tag, k):
    return Path(decode_root) / strategy / f"sub-{sub_label}" / tag / f"test_anova_{k}"


def load_fold_scores(decode_root, strategy, sub_label, tag="decode", k=1500):
    """Read cvscores.npy + fold_sizes.json written by decode.save_results()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    scores = np.load(condition_dir / "cvscores.npy")
    fold_sizes = json.loads((condition_dir / "fold_sizes.json").read_text())
    n_train = [fs[0] for fs in fold_sizes]
    n_test = [fs[1] for fs in fold_sizes]
    return scores, n_train, n_test


def load_predictions(decode_root, strategy, sub_label, tag="decode", k=1500):
    """Read y.npy + cv_predict.npy written by decode.save_results()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    y_true = np.load(condition_dir / "y.npy", allow_pickle=True)
    y_pred = np.load(condition_dir / "cv_predict.npy", allow_pickle=True)
    return y_true, y_pred


def weight_maps_path(decode_root, strategy, sub_label, tag="decode", k=1500):
    """Path to the 4D SVM weight-map NIfTI, written by decode.save_weight_maps()."""
    return (
        _condition_dir(decode_root, strategy, sub_label, tag, k) / "weight_maps.nii.gz"
    )


def weight_map_for_label(decode_root, strategy, sub_label, label, tag="decode", k=1500):
    """Load one class's averaged SVM weight map (a 3D volume) out of the 4D
    weight_maps.nii.gz + weight_maps_labels.json written by decode.save_weight_maps()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    labels = json.loads((condition_dir / "weight_maps_labels.json").read_text())
    img_4d = load_img(condition_dir / "weight_maps.nii.gz")
    return index_img(img_4d, labels.index(label))


def norm_map_path(
    decode_root, strategy, sub_label, tag="decode", k=1500, kind="weight"
):
    """Path to the per-subject voxel-importance map (norm across classes),
    written by decode.save_norm_map() as `<kind>_map_norm.nii.gz`."""
    return (
        _condition_dir(decode_root, strategy, sub_label, tag, k)
        / f"{kind}_map_norm.nii.gz"
    )


def load_group_size_scores(decode_root, strategy, sub_label, p, tag="cv_sweep", k=1500):
    """Read scores_p<p>.npy written by decode.save_group_size_comparison()."""
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    return np.load(condition_dir / f"scores_p{p}.npy")


def build_grid_long_dataframe(
    decode_root,
    strategies,
    subjects,
    ks,
    ps=(1, 2, 3, 4, 5),
    tag_main="decode",
    tag_sweep="cv_sweep",
    n_runs=6,
):
    """One row per (strategy, subject, k, p, fold): accuracy — input to the (k, p) grid.

    `p=1` is read from `tag_main`'s cvscores.npy, never from a `tag_sweep`
    run — `LeavePGroupsOut(1)` is mathematically identical to the
    `LeaveOneGroupOut` used by the main decode, so a p=1 cv_sweep run would
    be pure duplicate compute (see `decode.compare_group_sizes`).
    """
    rows = []
    for strategy in strategies:
        for sub_label in subjects:
            for k in ks:
                for p in ps:
                    if p == 1:
                        scores, _, _ = load_fold_scores(
                            decode_root, strategy, sub_label, tag_main, k
                        )
                    else:
                        scores = load_group_size_scores(
                            decode_root, strategy, sub_label, p, tag_sweep, k
                        )
                    for fold_idx, acc in enumerate(scores):
                        rows.append(
                            dict(
                                strategy=strategy,
                                subject=sub_label,
                                k=k,
                                p=p,
                                n_train_runs=n_runs - p,
                                fold=fold_idx,
                                accuracy=float(acc),
                            )
                        )
    return pd.DataFrame(rows)


def grid_summary(df_grid):
    """One row per (strategy, k, p): mean/std/sem/n_folds — for the grid heatmap/CSV."""
    summary = df_grid.groupby(["strategy", "k", "p", "n_train_runs"], as_index=False)[
        "accuracy"
    ].agg(mean="mean", std="std", n_folds="count")
    summary["sem"] = summary["std"] / np.sqrt(summary["n_folds"])
    return summary


def subject_level_pairwise_ttests(
    decode_root, strategies, subjects, tag="decode", k=1500
):
    """One row per (subject, strategy pair): means, Nadeau-Bengio t/p."""
    rows = []
    for sub_label in subjects:
        per_strategy = {
            s: load_fold_scores(decode_root, s, sub_label, tag, k) for s in strategies
        }
        for i, a in enumerate(strategies):
            for b in strategies[i + 1 :]:
                scores_a, n_train_a, n_test_a = per_strategy[a]
                scores_b, n_train_b, n_test_b = per_strategy[b]
                if len(scores_a) != len(scores_b):
                    raise ValueError(
                        f"sub-{sub_label}: {a} has {len(scores_a)} folds, {b} has "
                        f"{len(scores_b)} — a paired test requires the same number of folds."
                    )
                n_train = np.mean(n_train_a + n_train_b)
                n_test = np.mean(n_test_a + n_test_b)
                t_stat, p_value = nadeau_bengio_corrected_ttest(
                    scores_a, scores_b, n_train, n_test
                )
                rows.append(
                    dict(
                        subject=sub_label,
                        strategy_a=a,
                        strategy_b=b,
                        mean_a=float(scores_a.mean()),
                        mean_b=float(scores_b.mean()),
                        t_stat=t_stat,
                        p_value=p_value,
                    )
                )
    return pd.DataFrame(rows)


def build_long_dataframe(decode_root, strategies, subjects, tag="decode", k=1500):
    """One row per subject x fold x strategy — input to the group-level LMM."""
    rows = []
    for sub_label in subjects:
        for strategy in strategies:
            scores, _, _ = load_fold_scores(decode_root, strategy, sub_label, tag, k)
            for fold_idx, acc in enumerate(scores):
                rows.append(
                    dict(
                        subject=sub_label,
                        fold=fold_idx,
                        strategy=strategy,
                        accuracy=float(acc),
                    )
                )
    return pd.DataFrame(rows)


def fit_group_lmm(df_long, reference="fmriprep"):
    """accuracy ~ strategy (3-level fixed effect), subject as random effect
    (intercept), fitted on the fold-level data (not subject means).

    A random slope on strategy was tried but doesn't converge for this
    dataset's group count (subject-level sample size) — the slope variance
    estimates to the parameter-space boundary (~0), so the simpler
    random-intercept model is used instead.
    """
    formula = f'accuracy ~ C(strategy, Treatment(reference="{reference}"))'
    return smf.mixedlm(formula, df_long, groups=df_long["subject"]).fit()


def pairwise_contrasts_from_lmm(lmm_result, strategies, reference):
    """All pairwise strategy contrasts (n choose 2, for n strategies) from the ONE fitted model.

    Uses numeric contrast vectors (not string formulas) so that strategy names
    containing "+" (e.g. "fmriprep+denoised") can't be misparsed as arithmetic
    operators.
    """
    others = [s for s in strategies if s != reference]
    # fe_params (not params) — params also includes random-effect variance terms
    # (e.g. "Group Var"), but t_test's r_matrix must have exactly k_fe columns.
    param_names = list(lmm_result.fe_params.index)

    def term_for(level):
        return f'C(strategy, Treatment(reference="{reference}"))[T.{level}]'

    def contrast_row(strategy_a, strategy_b, coeffs):
        r = np.zeros((1, len(param_names)))
        for level, sign in coeffs:
            r[0, param_names.index(term_for(level))] = sign
        result = lmm_result.t_test(r)
        return dict(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            estimate=float(np.asarray(result.effect).ravel()[0]),
            p_value=float(np.asarray(result.pvalue).ravel()[0]),
        )

    rows = [contrast_row(reference, other, [(other, 1.0)]) for other in others]
    rows.extend(
        contrast_row(a, b, [(a, 1.0), (b, -1.0)]) for a, b in combinations(others, 2)
    )
    return pd.DataFrame(rows)


def hierarchical_bootstrap_contrast(
    df_long, strategy_a, strategy_b, n_boot=5000, seed=0
):
    """Two-stage (subject, then fold) bootstrap for the strategy_b - strategy_a
    group difference on the fold-level data — a nonparametric counterpart to
    `fit_group_lmm` that needs no REML convergence, useful as a check when the
    LMM lands on a boundary solution (see `fit_group_lmm`'s docstring).

    Each replicate resamples subjects with replacement, then independently
    resamples folds (with replacement) within each resampled subject, so both
    the between- and within-subject variance components feed the CI/p-value,
    same two levels the LMM's random intercept + residual represent.
    """
    rng = np.random.default_rng(seed)
    per_subject = {
        subject: (
            group[group["strategy"] == strategy_a]
            .sort_values("fold")["accuracy"]
            .to_numpy(),
            group[group["strategy"] == strategy_b]
            .sort_values("fold")["accuracy"]
            .to_numpy(),
        )
        for subject, group in df_long.groupby("subject")
    }
    subjects = np.array(list(per_subject))

    def group_stat(subject_sample):
        subject_diffs = []
        for subject in subject_sample:
            scores_a, scores_b = per_subject[subject]
            fold_idx = rng.integers(0, len(scores_a), len(scores_a))
            subject_diffs.append((scores_b[fold_idx] - scores_a[fold_idx]).mean())
        return float(np.mean(subject_diffs))

    observed = group_stat(subjects)
    boot_stats = np.array(
        [
            group_stat(rng.choice(subjects, size=len(subjects), replace=True))
            for _ in range(n_boot)
        ]
    )
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    p_value = min(2 * min((boot_stats <= 0).mean(), (boot_stats >= 0).mean()), 1.0)
    return dict(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        estimate=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
    )


def bootstrap_pairwise_contrasts(df_long, strategies, reference, n_boot=5000, seed=0):
    """All pairwise strategy contrasts via `hierarchical_bootstrap_contrast` — the
    nonparametric counterpart to `pairwise_contrasts_from_lmm`.

    Sign convention: `estimate = strategy_b - strategy_a` throughout (positive
    means strategy_b scores higher), so reference rows read as
    `other - reference`, matching the `gap = other - reference` convention
    used elsewhere (see `plot_grid_heatmap`).
    """
    others = [s for s in strategies if s != reference]
    rows = [
        hierarchical_bootstrap_contrast(df_long, reference, other, n_boot, seed)
        for other in others
    ]
    rows.extend(
        hierarchical_bootstrap_contrast(df_long, a, b, n_boot, seed)
        for a, b in combinations(others, 2)
    )
    return pd.DataFrame(rows)
