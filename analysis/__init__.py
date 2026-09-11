"""Consolidated statistics + figures comparing denoising strategies.

Subject-level: paired t-test with the Nadeau-Bengio correction for CV fold
correlation (see `stats.nadeau_bengio_corrected_ttest`), applied to every
pairwise strategy comparison, per subject.

Group-level: a single linear mixed-effects model (`accuracy ~ strategy`,
subject as random effect) fitted once on the fold-level data (not subject
means), with all 3 pairwise strategy contrasts derived from that one fit (see
`stats.pairwise_contrasts_from_lmm`) — no repeated fitting, no extra
multiple-comparison correction needed since these are contrasts of one model
rather than independent tests. A hierarchical (subject-then-fold) bootstrap
(`stats.bootstrap_pairwise_contrasts`) provides a nonparametric check on the
same contrasts, since it needs no REML convergence.
"""

from pathlib import Path

from _constants import STRATEGIES, SUBJECTS
from analysis.plotting import (
    plot_grid_heatmap,
    plot_grid_lines,
    plot_group_comparison,
    plot_subject_accuracy,
    plot_weight_map_comparison,
    save_confusion_matrix,
)
from analysis.stats import (
    bootstrap_pairwise_contrasts,
    build_grid_long_dataframe,
    build_long_dataframe,
    fit_group_lmm,
    grid_summary,
    load_predictions,
    pairwise_contrasts_from_lmm,
    subject_level_pairwise_ttests,
    weight_map_for_label,
    weight_maps_path,
)
from glm import find_t1w


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
):
    """Run subject-level and group-level statistics + figures for `strategies`.

    `decode_root` must contain one subdirectory per strategy (as produced by
    running `decode --output-dir <decode_root>/<strategy>` once per strategy),
    each holding `sub-XX/<tag>/test_anova_<k>/...` outputs from `decode.save_results`.

    `dataset_dir`/`mask_derivatives_folder` locate each subject's native-space
    T1w (see `glm.find_t1w`) — always the plain fMRIPrep derivatives, same
    convention as `find_brain_mask`, used as the background for weight-map
    figures.
    """
    strategies = list(strategies)
    output_dir = Path(output_dir)

    subject_stats = subject_level_pairwise_ttests(
        decode_root, strategies, subjects, tag, k
    )

    for sub_label in subjects:
        sub_dir = output_dir / "subjects" / f"sub-{sub_label}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        rows = subject_stats[subject_stats["subject"] == sub_label]
        rows.to_csv(sub_dir / "pairwise_ttests.csv", index=False)

        sub_long = build_long_dataframe(decode_root, strategies, [sub_label], tag, k)
        plot_subject_accuracy(
            sub_long, rows, sub_label, sub_dir / "accuracy_boxplot.png"
        )

        for strategy in strategies:
            y_true, y_pred = load_predictions(decode_root, strategy, sub_label, tag, k)
            labels = sorted(set(y_true) | set(y_pred))
            save_confusion_matrix(
                y_true,
                y_pred,
                labels,
                sub_dir / f"confusion_matrix_{strategy}.png",
                title=f"sub-{sub_label} — {strategy}",
            )

        # Weight maps compare denoising's effect on what the SVM learned, per
        # condition. Guarded by exists() since older decode outputs (before
        # decode.save_weight_maps existed) won't have these files.
        have_weight_maps = all(
            weight_maps_path(decode_root, strategy, sub_label, tag, k).exists()
            for strategy in strategies
        )
        if have_weight_maps:
            bg_img = find_t1w(dataset_dir, sub_label, mask_derivatives_folder)
            for label in labels:
                imgs = {
                    strategy: weight_map_for_label(
                        decode_root, strategy, sub_label, label, tag, k
                    )
                    for strategy in strategies
                }
                plot_weight_map_comparison(
                    imgs,
                    bg_img,
                    sub_dir / f"weight_map_{label}.png",
                    title=f"sub-{sub_label} — {label}",
                )

    df_long = build_long_dataframe(decode_root, strategies, subjects, tag, k)
    lmm_result = fit_group_lmm(df_long, reference=reference_strategy)
    contrasts = pairwise_contrasts_from_lmm(lmm_result, strategies, reference_strategy)
    bootstrap_contrasts = bootstrap_pairwise_contrasts(
        df_long, strategies, reference_strategy
    )

    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "lmm_summary.txt").write_text(str(lmm_result.summary()))
    contrasts.to_csv(group_dir / "pairwise_contrasts.csv", index=False)
    bootstrap_contrasts.to_csv(
        group_dir / "bootstrap_pairwise_contrasts.csv", index=False
    )
    subject_stats.to_csv(group_dir / "subject_level_pairwise_ttests.csv", index=False)
    plot_group_comparison(df_long, contrasts, group_dir / "group_comparison.png")

    return lmm_result, contrasts, bootstrap_contrasts, subject_stats


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
        output_dir / "heatmap_accuracy.png",
        reference_strategy=reference_strategy,
    )
    plot_grid_lines(df_summary, strategies, output_dir / "lines_accuracy_vs_k.png")

    return df_long, df_summary
