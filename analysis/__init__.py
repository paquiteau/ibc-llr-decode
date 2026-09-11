"""Consolidated statistics + figures comparing denoising strategies.

Subject-level: paired t-test with the Nadeau-Bengio correction for CV fold
correlation (see `stats.nadeau_bengio_corrected_ttest`), applied to every
pairwise strategy comparison, per subject.

Group-level: a single linear mixed-effects model (`accuracy ~ strategy`,
subject as random effect) fitted once, with all 3 pairwise strategy contrasts
derived from that one fit (see `stats.pairwise_contrasts_from_lmm`) — no
repeated fitting, no extra multiple-comparison correction needed since these
are contrasts of one model rather than independent tests.
"""

from pathlib import Path

from _constants import STRATEGIES, SUBJECTS
from analysis.plotting import (
    plot_group_comparison,
    plot_subject_accuracy,
    save_confusion_matrix,
)
from analysis.stats import (
    build_long_dataframe,
    fit_group_lmm,
    load_predictions,
    pairwise_contrasts_from_lmm,
    subject_level_pairwise_ttests,
)


def run_analysis(
    decode_root,
    output_dir,
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

    df_long = build_long_dataframe(decode_root, strategies, subjects, tag, k)
    lmm_result = fit_group_lmm(df_long, reference=reference_strategy)
    contrasts = pairwise_contrasts_from_lmm(lmm_result, strategies, reference_strategy)

    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "lmm_summary.txt").write_text(str(lmm_result.summary()))
    contrasts.to_csv(group_dir / "pairwise_contrasts.csv", index=False)
    subject_stats.to_csv(group_dir / "subject_level_pairwise_ttests.csv", index=False)
    plot_group_comparison(df_long, contrasts, group_dir / "group_comparison.png")

    return lmm_result, contrasts, subject_stats
