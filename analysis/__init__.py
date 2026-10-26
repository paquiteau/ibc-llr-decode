"""Statistics + figures comparing preprocessing strategies.

All tests compare each strategy to a reference on paired values, one per
subject (summary-statistic approach, see `stats`' module docstring):

- Group level: exact sign-flip permutation test on the per-subject accuracy
  differences (primary p-value), with the paired t-test's 95% CI and Cohen's
  d_z as effect sizes, Holm-corrected across comparisons.
- Training size: the same test at each number of held-out runs p (Holm across
  p), plus a test on per-subject slopes of the gain vs. training runs.
- Single subject: exact sign-flip test on the paired out-of-fold accuracy
  differences of mini-blocks (trials when `block_size=1`, where it is the
  exact McNemar test), BH-FDR across subjects.
- Per class: the group test on per-class recall differences, Holm across
  classes.
"""

from pathlib import Path

import numpy as np

from _constants import CONDITION_LABELS, STRATEGIES, SUBJECTS
from analysis.plotting import (
    plot_confusion_matrices,
    plot_decoding_summary,
    plot_grid_heatmap,
    plot_grid_lines,
    plot_group_accuracy,
    plot_norm_map_comparison,
    plot_subject_folds,
    plot_training_size,
    plot_weight_map_comparison,
    strategy_label,
)
from analysis.stats import (
    build_grid_long_dataframe,
    build_long_dataframe,
    fold_predictions_path,
    grid_summary,
    group_contrasts,
    group_size_scores_path,
    load_fold_predictions,
    mean_ci,
    norm_map_path,
    per_class_contrasts,
    subject_block_tests,
    subject_confusion_matrices,
    subject_means,
    training_size_contrasts,
    training_size_trend,
    weight_map_for_label,
    weight_maps_path,
)
from glm import find_t1w


def ordered_labels(labels):
    """Condition labels in `CONDITION_LABELS` order, unknown ones last."""
    labels = set(labels)
    return [c for c in CONDITION_LABELS if c in labels] + sorted(
        labels - set(CONDITION_LABELS)
    )


def run_analysis(
    decode_root,
    output_dir,
    dataset_dir,
    mask_derivatives_folder,
    strategies=STRATEGIES,
    subjects=SUBJECTS,
    tag="decode",
    k=1500,
    reference_strategy="fmriprep",
    ps=(1, 2, 3, 4),
    tag_sweep="cv_sweep",
    block_size=1,
):
    """Run group/subject-level statistics + figures for `strategies` vs `reference_strategy`.

    `decode_root` must contain one subdirectory per strategy (as produced by
    running `decode --output-dir <decode_root>/<strategy>` once per strategy),
    each holding `sub-XX/<tag>/test_anova_<k>/...` outputs from `decode.save_results`.
    The training-size analysis additionally reads the `<tag_sweep>` outputs
    of `decode --leave-p-out` for every p > 1 in `ps`, and is skipped if any
    are missing.

    `dataset_dir`/`mask_derivatives_folder` locate each subject's native-space
    T1w (see `glm.find_t1w`) — always the plain fMRIPrep derivatives, same
    convention as `find_brain_mask`, used as the background for weight-map
    figures.

    `block_size` sets the unit of the single-subject test (see
    `stats.block_units`). Chance is 1 / number of classes (balanced designs)
    and the number of runs is read from the CV groups.
    """
    others = [s for s in strategies if s != reference_strategy]
    strategies = [reference_strategy, *others]
    subjects = list(subjects)
    output_dir = Path(output_dir)
    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    y_true, groups, _ = load_fold_predictions(
        decode_root, reference_strategy, subjects[0], 1, tag, tag_sweep, k
    )
    labels = ordered_labels(y_true)
    chance = 1 / len(labels)
    n_runs = len(np.unique(groups))

    df_long = build_long_dataframe(decode_root, strategies, subjects, tag, k)
    means = subject_means(df_long)[strategies]
    contrasts = group_contrasts(means, reference_strategy, others, chance)

    # p > 1 single-subject tests need every fold's predictions, only written
    # by decode runs recent enough to save them.
    test_ps = [1] + [
        p
        for p in ps
        if p > 1
        and all(
            fold_predictions_path(decode_root, s, sub, p, tag_sweep, k).exists()
            for s in strategies
            for sub in subjects
        )
    ]
    subject_tests = subject_block_tests(
        decode_root,
        reference_strategy,
        others,
        subjects,
        block_size,
        test_ps,
        tag,
        tag_sweep,
        k,
    )

    confusions = subject_confusion_matrices(
        decode_root, strategies, subjects, labels, tag, k
    )
    per_class = per_class_contrasts(confusions, labels, reference_strategy, others)

    means.to_csv(group_dir / "subject_accuracy.csv")
    contrasts.to_csv(group_dir / "group_contrasts.csv", index=False)
    subject_tests.to_csv(group_dir / "subject_tests.csv", index=False)
    per_class.to_csv(group_dir / "per_class_recall.csv", index=False)
    plot_group_accuracy(
        means, contrasts, reference_strategy, group_dir / "fig_accuracy.pdf", chance
    )
    plot_confusion_matrices(
        {s: m.mean(axis=0) for s, m in confusions.items()},
        labels,
        reference_strategy,
        group_dir / "fig_confusion.pdf",
        per_class=per_class,
        png=True,
    )

    sweep_ps = [p for p in ps if p > 1]
    have_sweep = sweep_ps and all(
        group_size_scores_path(decode_root, s, sub, p, tag_sweep, k).exists()
        for s in strategies
        for sub in subjects
        for p in sweep_ps
    )
    size_contrasts = size_trend = None
    if have_sweep:
        df_grid = build_grid_long_dataframe(
            decode_root, strategies, subjects, [k], ps, tag, tag_sweep, n_runs
        )
        size_contrasts = training_size_contrasts(df_grid, reference_strategy, others)
        size_trend = training_size_trend(df_grid, reference_strategy, others)
        size_contrasts.to_csv(group_dir / "training_size_contrasts.csv", index=False)
        size_trend.to_csv(group_dir / "training_size_trend.csv", index=False)
        plot_training_size(
            df_grid,
            size_contrasts,
            reference_strategy,
            group_dir / "fig_training_size.pdf",
        )
        plot_decoding_summary(
            means,
            df_grid,
            size_contrasts,
            confusions,
            labels,
            per_class,
            reference_strategy,
            group_dir / "fig_decoding_summary.pdf",
            chance,
        )
    elif sweep_ps:
        print(
            f"Missing {tag_sweep} outputs for p={sweep_ps}: skipping training-size analysis."
        )

    summary = _summary_text(
        means,
        contrasts,
        subject_tests,
        per_class,
        size_contrasts,
        size_trend,
        k,
        chance,
        block_size,
    )
    (group_dir / "summary.txt").write_text(summary)
    print(summary)

    for sub_label in subjects:
        _subject_figures(
            decode_root,
            output_dir / "subjects" / f"sub-{sub_label}",
            dataset_dir,
            mask_derivatives_folder,
            strategies,
            sub_label,
            labels,
            df_long[df_long["subject"] == sub_label],
            subject_tests[subject_tests["subject"] == sub_label],
            {s: m[subjects.index(sub_label)] for s, m in confusions.items()},
            tag,
            k,
            chance,
        )

    return contrasts, subject_tests, per_class, size_contrasts, size_trend


def _subject_figures(
    decode_root,
    sub_dir,
    dataset_dir,
    mask_derivatives_folder,
    strategies,
    sub_label,
    labels,
    sub_long,
    sub_tests,
    sub_confusions,
    tag,
    k,
    chance,
):
    reference = strategies[0]
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_tests.to_csv(sub_dir / "tests.csv", index=False)
    plot_subject_folds(
        sub_long,
        sub_tests[sub_tests["p"] == 1],
        reference,
        sub_label,
        sub_dir / "accuracy_folds.pdf",
        chance,
    )
    plot_confusion_matrices(
        sub_confusions, labels, reference, sub_dir / "confusion_matrices.pdf"
    )

    # Weight maps compare denoising's effect on what the SVM learned, per
    # condition. Guarded by exists() since older decode outputs (before
    # decode.save_weight_maps existed) won't have these files.
    bg_img = find_t1w(dataset_dir, sub_label, mask_derivatives_folder)
    if all(
        weight_maps_path(decode_root, s, sub_label, tag, k).exists() for s in strategies
    ):
        for label in labels:
            imgs = {
                s: weight_map_for_label(decode_root, s, sub_label, label, tag, k)
                for s in strategies
            }
            plot_weight_map_comparison(
                imgs,
                bg_img,
                sub_dir / f"weight_map_{label}.pdf",
                title=f"sub-{sub_label}, {label}",
            )

    # Per-subject voxel-importance maps (norm across classes) — how much
    # each voxel matters for decoding, independent of any one condition.
    # Raw-weight norm can overweight noise-cancelling voxels (Haufe et
    # al., 2014); the pattern-norm map is the activation-pattern
    # counterpart.
    for kind in ("weight", "pattern"):
        paths = {
            s: norm_map_path(decode_root, s, sub_label, tag, k, kind)
            for s in strategies
        }
        if all(path.exists() for path in paths.values()):
            plot_norm_map_comparison(
                paths,
                bg_img,
                sub_dir / f"{kind}_map_norm.pdf",
                title=f"sub-{sub_label}, {kind} norm",
            )


def _summary_text(
    means,
    contrasts,
    subject_tests,
    per_class,
    size_contrasts,
    size_trend,
    k,
    chance,
    block_size,
):
    pct = 100
    lines = [
        f"Decoding accuracy, k={k}, leave-one-run-out CV, {len(means)} subjects "
        f"(chance {pct * chance:.1f}%). Mean [95% CI], range over subjects:"
    ]
    for strategy in means.columns:
        mean, low, high = mean_ci(means[strategy])
        lines.append(
            f"  {strategy_label(strategy)}: {pct * mean:.1f}% [{pct * low:.1f}, "
            f"{pct * high:.1f}], {pct * means[strategy].min():.1f}-"
            f"{pct * means[strategy].max():.1f}%"
        )
    lines.append(
        "\nGroup contrasts (exact sign-flip permutation test on per-subject "
        "differences; t-based 95% CI; Holm across comparisons):"
    )
    for r in contrasts.itertuples():
        lines.append(
            f"  {strategy_label(r.strategy_b)} - {strategy_label(r.strategy_a)}: "
            f"{pct * r.diff:+.2f} pp [{pct * r.ci_low:+.2f}, {pct * r.ci_high:+.2f}], "
            f"p_perm = {r.p_perm:.2g} (Holm {r.p_holm:.2g}), t({r.df}) = {r.t:.2f}, "
            f"p_t = {r.p_ttest:.2g}, d_z = {r.cohen_dz:.2f}, improved in "
            f"{r.n_improved}/{r.n_subjects} subjects, above-chance accuracy "
            f"{pct * r.gain_above_chance:+.1f}%"
        )
    unit = "trials" if block_size == 1 else f"mini-blocks of {block_size} trials"
    lines.append(
        f"\nSingle subject (exact sign-flip test over {unit}, BH-FDR across subjects):"
    )
    for (other, n_train), rows in subject_tests.groupby(
        ["strategy_b", "n_train_runs"], sort=False
    ):
        lines.append(
            f"  {strategy_label(other)}, {n_train} training runs: q < 0.05 in "
            f"{(rows['q_fdr'] < 0.05).sum()}/{len(rows)} subjects (all improvements: "
            f"{((rows['q_fdr'] < 0.05) & (rows['diff'] > 0)).sum()}), "
            f"uncorrected p < 0.05 in {(rows['p_signflip'] < 0.05).sum()}"
        )
        for r in rows.itertuples():
            lines.append(
                f"    sub-{r.subject}: {pct * r.acc_a:.1f}% -> {pct * r.acc_b:.1f}% "
                f"({pct * r.diff:+.2f} pp), {r.units_b_better} better / "
                f"{r.units_a_better} worse of {r.n_units} units, "
                f"p = {r.p_signflip:.2g}, q = {r.q_fdr:.2g}"
            )
    lines.append("\nPer-class recall difference (Holm across classes):")
    for r in per_class.itertuples():
        lines.append(
            f"  {strategy_label(r.strategy_b)}, {r.label}: {pct * r.diff:+.2f} pp "
            f"[{pct * r.ci_low:+.2f}, {pct * r.ci_high:+.2f}], p_perm = {r.p_perm:.2g} "
            f"(Holm {r.p_holm:.2g})"
        )
    if size_contrasts is not None:
        lines.append(
            "\nTraining size (same group test per number of training runs; Holm across p):"
        )
        for r in size_contrasts.itertuples():
            lines.append(
                f"  {strategy_label(r.strategy_b)}, {r.n_train_runs} training runs: "
                f"{pct * r.mean_a:.1f}% -> {pct * r.mean_b:.1f}%, {pct * r.diff:+.2f} pp "
                f"[{pct * r.ci_low:+.2f}, {pct * r.ci_high:+.2f}], p_perm = {r.p_perm:.2g} "
                f"(Holm {r.p_holm:.2g}), improved in {r.n_improved}/{r.n_subjects}"
            )
        for r in size_trend.itertuples():
            lines.append(
                f"  Trend of the gain per additional training run "
                f"({strategy_label(r.strategy_b)}): {pct * r.slope:+.2f} pp/run "
                f"[{pct * r.ci_low:+.2f}, {pct * r.ci_high:+.2f}], p_perm = {r.p_perm:.2g}, "
                f"negative in {r.n_negative}/{r.n_subjects} subjects"
            )
    return "\n".join(lines) + "\n"


def run_grid_analysis(
    decode_root,
    output_dir,
    strategies,
    subjects,
    ks,
    ps,
    tag_main="decode",
    tag_sweep="cv_sweep",
    reference_strategy="fmriprep",
):
    """Aggregate a (k, p) grid search into a long dataframe + heatmap/line figures.

    p=1 is read from `tag_main`'s decode output, never from `tag_sweep` — see
    `stats.build_grid_long_dataframe`. Answers "which (k, p) region looks
    good", distinct from `run_analysis`'s fixed-k strategy comparison.
    """
    strategies = list(strategies)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_long = build_grid_long_dataframe(
        decode_root, strategies, subjects, ks, ps, tag_main, tag_sweep
    )
    df_summary = grid_summary(df_long)

    df_long.to_csv(output_dir / "grid_long.csv", index=False)
    df_summary.to_csv(output_dir / "grid_summary.csv", index=False)
    plot_grid_heatmap(
        df_summary,
        strategies,
        output_dir / "heatmap_accuracy.pdf",
        reference_strategy=reference_strategy,
    )
    plot_grid_lines(df_summary, strategies, output_dir / "lines_accuracy_vs_k.pdf")

    return df_long, df_summary
