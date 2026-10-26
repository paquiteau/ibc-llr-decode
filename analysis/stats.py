"""Subject- and group-level statistics for comparing preprocessing strategies.

The design is fully paired and balanced: every strategy is decoded on the same
trials with the same CV folds, and every subject has the same number of runs
and trials. Population inference therefore uses the summary-statistic approach
(Holmes & Friston, 1998): one value per subject and strategy (accuracy averaged
over folds), and a test across subjects on the paired differences. For a
balanced design this is equivalent to a mixed model with a random subject slope
on strategy, without its convergence issues, and it stays valid when CV folds
overlap (`LeavePGroupsOut` with p > 1), since folds are never treated as
independent observations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.image import index_img, load_img
from scipy import stats
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests


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


def load_fold_predictions(
    decode_root, strategy, sub_label, p=1, tag="decode", tag_sweep="cv_sweep", k=1500
):
    """(y, g, predictions) of one subject: `predictions` is (n_folds, n_trials),
    "" where a trial was not tested in that fold.

    p=1 reads the main decode's `cv_predict.npy` (one row: every trial is
    tested exactly once); p > 1 reads `cv_predict_folds_p<p>.npy`, written by
    `decode.save_group_size_comparison` (`LeavePGroupsOut` tests each trial in
    several folds, so a single prediction per trial would drop all but one).
    """
    condition_dir = _condition_dir(decode_root, strategy, sub_label, tag, k)
    y = np.load(condition_dir / "y.npy", allow_pickle=True)
    g = np.load(condition_dir / "g.npy", allow_pickle=True)
    if p == 1:
        preds = np.load(condition_dir / "cv_predict.npy", allow_pickle=True)[None]
    else:
        sweep_dir = _condition_dir(decode_root, strategy, sub_label, tag_sweep, k)
        preds = np.load(sweep_dir / f"cv_predict_folds_p{p}.npy", allow_pickle=True)
    return y, g, preds


def fold_predictions_path(decode_root, strategy, sub_label, p, tag="cv_sweep", k=1500):
    return (
        _condition_dir(decode_root, strategy, sub_label, tag, k)
        / f"cv_predict_folds_p{p}.npy"
    )


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


def group_size_scores_path(decode_root, strategy, sub_label, p, tag="cv_sweep", k=1500):
    """Path to scores_p<p>.npy written by decode.save_group_size_comparison()."""
    return _condition_dir(decode_root, strategy, sub_label, tag, k) / f"scores_p{p}.npy"


def load_group_size_scores(decode_root, strategy, sub_label, p, tag="cv_sweep", k=1500):
    return np.load(group_size_scores_path(decode_root, strategy, sub_label, p, tag, k))


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


def build_long_dataframe(decode_root, strategies, subjects, tag="decode", k=1500):
    """One row per subject x fold x strategy."""
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


def subject_means(df_long):
    """subject x strategy table of accuracy averaged over CV folds.

    Folds have equal test sizes, so this is also the pooled trial accuracy.
    """
    return df_long.pivot_table(
        index="subject", columns="strategy", values="accuracy", aggfunc="mean"
    )


def mean_ci(values, confidence=0.95):
    """Mean and t-based confidence interval, as (mean, low, high)."""
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    low, high = stats.t.interval(
        confidence, values.size - 1, loc=mean, scale=stats.sem(values)
    )
    return float(mean), float(low), float(high)


def sign_flip_test(values, n_perm=100_000, seed=0):
    """Two-sided sign-flip permutation p-value for H0: mean(values) = 0.

    Under H0 (paired differences symmetric around zero), each value's sign is
    exchangeable. Exact (all 2**n sign patterns) for n <= 20, Monte Carlo with
    `n_perm` patterns (identity included) otherwise.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
        signs[0] = 1
    null = np.abs(signs @ values)
    return float(np.mean(null >= np.abs(values.sum()) * (1 - 1e-12)))


def paired_contrast(a, b):
    """Group-level `b - a` from one paired value per subject.

    `p_perm` (exact sign-flip test) is the primary p-value; the paired t-test
    gives the 95% CI and is reported alongside as its parametric counterpart.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b - a
    ttest = stats.ttest_1samp(diff, 0.0)
    ci = ttest.confidence_interval(0.95)
    return dict(
        n_subjects=diff.size,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=float(diff.mean()),
        ci_low=float(ci.low),
        ci_high=float(ci.high),
        t=float(ttest.statistic),
        df=int(ttest.df),
        p_ttest=float(ttest.pvalue),
        p_perm=sign_flip_test(diff),
        cohen_dz=float(diff.mean() / diff.std(ddof=1)),
        n_improved=int((diff > 0).sum()),
    )


def _holm(p_values):
    return multipletests(p_values, method="holm")[1]


def group_contrasts(means, reference, others, chance):
    """Each strategy in `others` vs `reference` on subject means (see `subject_means`),
    Holm-corrected across comparisons.

    `gain_above_chance` is the relative increase of above-chance accuracy,
    (mean_b - chance) / (mean_a - chance) - 1.
    """
    rows = [
        dict(
            strategy_a=reference,
            strategy_b=other,
            **paired_contrast(means[reference], means[other]),
        )
        for other in others
    ]
    df = pd.DataFrame(rows)
    df["gain_above_chance"] = (df["mean_b"] - chance) / (df["mean_a"] - chance) - 1
    df["p_holm"] = _holm(df["p_perm"])
    return df


def training_size_contrasts(df_grid, reference, others):
    """`group_contrasts` repeated per number of held-out runs `p`, Holm-corrected
    across `p` within each comparison. `df_grid` from `build_grid_long_dataframe`
    restricted to a single `k`."""
    rows = []
    for (p, n_train_runs), df_p in df_grid.groupby(["p", "n_train_runs"]):
        means = subject_means(df_p)
        for other in others:
            rows.append(
                dict(
                    p=p,
                    n_train_runs=n_train_runs,
                    strategy_a=reference,
                    strategy_b=other,
                    **paired_contrast(means[reference], means[other]),
                )
            )
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby("strategy_b")["p_perm"].transform(_holm)
    return df


def training_size_trend(df_grid, reference, others):
    """Does the gain depend on the amount of training data?

    Per subject, the OLS slope of the paired difference (`other - reference`)
    against the number of training runs; slopes are then tested across
    subjects like any paired difference (summary-statistic approach, so the
    overlap of `LeavePGroupsOut` folds within a subject doesn't matter). A
    negative slope means the gain shrinks as training data grows.
    """
    means = df_grid.pivot_table(
        index=["subject", "n_train_runs"],
        columns="strategy",
        values="accuracy",
        aggfunc="mean",
    )
    rows = []
    for other in others:
        diff = (means[other] - means[reference]).unstack("n_train_runs")
        x = diff.columns.to_numpy(dtype=float)
        slopes = np.polyfit(x, diff.to_numpy().T, 1)[0]
        contrast = paired_contrast(np.zeros_like(slopes), slopes)
        rows.append(
            dict(
                strategy_a=reference,
                strategy_b=other,
                n_subjects=contrast["n_subjects"],
                slope=contrast["diff"],
                ci_low=contrast["ci_low"],
                ci_high=contrast["ci_high"],
                t=contrast["t"],
                df=contrast["df"],
                p_ttest=contrast["p_ttest"],
                p_perm=contrast["p_perm"],
                n_negative=int((slopes < 0).sum()),
            )
        )
    return pd.DataFrame(rows)


def block_units(y, g, block_size=1):
    """Exchangeable unit of each trial for the single-subject test.

    A unit is `block_size` consecutive trials of the same condition in a run
    (a mini-block); `block_size=1` makes every trial its own unit. Trials of a
    (run, condition) are stored in time order (see `decode.load_glm_data`),
    so the i-th of them belongs to that condition's (i // block_size)-th block.
    """
    rank = pd.Series(np.arange(len(y))).groupby([g, y]).cumcount().to_numpy()
    keys = [f"{run}|{cond}|{r // block_size}" for run, cond, r in zip(g, y, rank)]
    return np.unique(keys, return_inverse=True)[1]


def exact_sign_flip_p(values):
    """Exact two-sided sign-flip p-value for H0: sum(values) = 0, integer `values`.

    Under H0 each value's sign is exchangeable, so the null distribution of
    the sum is the convolution of the two-point distributions {+v, -v}.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("exact_sign_flip_p needs integer-valued differences")
    observed = abs(int(values.sum()))
    magnitudes = np.abs(values[values != 0])
    total = int(magnitudes.sum())
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0
    for v in magnitudes:
        # the support stays within [-total, total], so np.roll never wraps mass
        dist = 0.5 * (np.roll(dist, v) + np.roll(dist, -v))
    support = np.arange(-total, total + 1)
    return float(min(1.0, dist[np.abs(support) >= observed].sum()))


def subject_block_tests(
    decode_root,
    reference,
    others,
    subjects,
    block_size=1,
    ps=(1,),
    tag="decode",
    tag_sweep="cv_sweep",
    k=1500,
):
    """Single-subject `other` vs `reference` on the same trials and CV folds.

    For each unit (see `block_units`), the paired difference is the number of
    its out-of-fold predictions correct for `other` minus for `reference`,
    summed over its trials and over every fold testing them; the test is an
    exact sign-flip test on those differences (`exact_sign_flip_p`), BH-FDR
    across subjects within each comparison and number of held-out runs `p`.

    With `block_size=1` and p=1 this is the exact McNemar test (only discordant
    trials differ, by +-1). Mini-blocks account for the dependence of trials a
    few seconds apart within a block; folds testing the same trial (p > 1)
    are summed within the unit, so they are never treated as independent.
    `n_train_runs` is the number of runs minus p.
    """
    rows = []
    for sub_label in subjects:
        for p in ps:
            y, g, preds_a = load_fold_predictions(
                decode_root, reference, sub_label, p, tag, tag_sweep, k
            )
            units = block_units(y, g, block_size)
            tested = preds_a != ""
            correct_a = (preds_a == y) & tested
            for other in others:
                y_b, g_b, preds_b = load_fold_predictions(
                    decode_root, other, sub_label, p, tag, tag_sweep, k
                )
                if not (
                    np.array_equal(y, y_b)
                    and np.array_equal(g, g_b)
                    and np.array_equal(tested, preds_b != "")
                ):
                    raise ValueError(
                        f"sub-{sub_label}, p={p}: {reference} and {other} were "
                        "decoded on different trials or folds — a paired test "
                        "requires the same ones."
                    )
                correct_b = (preds_b == y) & tested
                diff = (correct_b.astype(int) - correct_a).sum(axis=0)
                unit_diff = np.bincount(units, weights=diff).astype(int)
                rows.append(
                    dict(
                        subject=sub_label,
                        p=p,
                        n_train_runs=len(np.unique(g)) - p,
                        strategy_a=reference,
                        strategy_b=other,
                        acc_a=correct_a.sum() / tested.sum(),
                        acc_b=correct_b.sum() / tested.sum(),
                        diff=(correct_b.sum() - correct_a.sum()) / tested.sum(),
                        n_units=unit_diff.size,
                        units_b_better=int((unit_diff > 0).sum()),
                        units_a_better=int((unit_diff < 0).sum()),
                        p_signflip=exact_sign_flip_p(unit_diff),
                    )
                )
    df = pd.DataFrame(rows)
    df["q_fdr"] = df.groupby(["strategy_b", "p"])["p_signflip"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    return df


def subject_confusion_matrices(
    decode_root, strategies, subjects, labels, tag="decode", k=1500
):
    """{strategy: (n_subjects, n_classes, n_classes)} row-normalized confusion
    matrices (rows: true class, so the diagonal is per-class recall)."""
    return {
        strategy: np.stack(
            [
                confusion_matrix(
                    *load_predictions(decode_root, strategy, sub_label, tag, k),
                    labels=labels,
                    normalize="true",
                )
                for sub_label in subjects
            ]
        )
        for strategy in strategies
    }


def per_class_contrasts(confusions, labels, reference, others):
    """Per-class recall `other - reference` across subjects, Holm-corrected
    across classes within each comparison."""
    rows = []
    for other in others:
        for i, label in enumerate(labels):
            rows.append(
                dict(
                    label=label,
                    strategy_a=reference,
                    strategy_b=other,
                    **paired_contrast(
                        confusions[reference][:, i, i], confusions[other][:, i, i]
                    ),
                )
            )
    df = pd.DataFrame(rows)
    df["p_holm"] = df.groupby("strategy_b")["p_perm"].transform(_holm)
    return df
