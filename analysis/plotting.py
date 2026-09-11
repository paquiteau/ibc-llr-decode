"""Shared plotting primitives — one implementation each, reused by subject- and group-level figures."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from _constants import CHANCE


def stars_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_significance_brackets(ax, categories, pairwise, y_start, step):
    for i, row in enumerate(pairwise.itertuples()):
        x1, x2 = categories.index(row.strategy_a), categories.index(row.strategy_b)
        y = y_start + step * (i + 1)
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + step / 4, y + step / 4, y],
            color="black",
            linewidth=1,
        )
        ax.text(
            (x1 + x2) / 2,
            y + step / 4,
            stars_from_p(row.p_value),
            ha="center",
            va="bottom",
        )
    return y_start + step * (len(pairwise) + 1.5)


def plot_confusion_matrix(
    y_true, y_pred, labels, ax=None, normalize="true", title=None
):
    """Single canonical confusion-matrix plotter (replaces 3 independent copies)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    ax = ax or plt.gca()
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    return ax


def save_confusion_matrix(y_true, y_pred, labels, output_path, title=None):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    plot_confusion_matrix(y_true, y_pred, labels, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_subject_accuracy(df_subject_long, pairwise_stats, subject, output_path):
    """Per-subject box/strip of per-fold accuracy across strategies, with chance line
    and significance stars from `subject_level_pairwise_ttests`."""
    strategies = list(df_subject_long["strategy"].unique())
    fig, ax = plt.subplots(figsize=(1.5 * len(strategies) + 2, 4))
    sns.boxplot(
        data=df_subject_long,
        x="strategy",
        y="accuracy",
        order=strategies,
        ax=ax,
        showfliers=False,
    )
    sns.stripplot(
        data=df_subject_long,
        x="strategy",
        y="accuracy",
        order=strategies,
        ax=ax,
        color="black",
        alpha=0.6,
    )
    ax.axhline(CHANCE, ls="--", color="gray", label="chance")

    y_top = _add_significance_brackets(
        ax, strategies, pairwise_stats, df_subject_long["accuracy"].max(), step=0.05
    )
    ax.set_ylim(top=max(ax.get_ylim()[1], y_top))
    ax.set_title(f"sub-{subject}")
    ax.set_ylabel("Decoding accuracy")
    ax.set_xlabel("")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_comparison(df_long, pairwise_contrasts, output_path):
    """Group-level box/strip: one point per subject per strategy (mean over folds),
    subjects connected across strategies, significance bars from the fitted LMM contrasts."""
    strategies = list(df_long["strategy"].unique())
    means = df_long.groupby(["subject", "strategy"], as_index=False)["accuracy"].mean()

    fig, ax = plt.subplots(figsize=(1.5 * len(strategies) + 2, 5))
    sns.boxplot(
        data=means,
        x="strategy",
        y="accuracy",
        order=strategies,
        ax=ax,
        showfliers=False,
        color="lightgray",
    )
    sns.stripplot(
        data=means,
        x="strategy",
        y="accuracy",
        order=strategies,
        ax=ax,
        hue="subject",
        palette="tab20",
        legend=False,
    )
    for _, sub_df in means.groupby("subject"):
        sub_df = sub_df.set_index("strategy").reindex(strategies)
        ax.plot(
            range(len(strategies)),
            sub_df["accuracy"],
            color="gray",
            alpha=0.3,
            linewidth=0.8,
        )
    ax.axhline(CHANCE, ls="--", color="gray", label="chance")

    y_top = _add_significance_brackets(
        ax, strategies, pairwise_contrasts, means["accuracy"].max(), step=0.03
    )
    ax.set_ylim(top=max(ax.get_ylim()[1], y_top))
    ax.set_ylabel("Mean decoding accuracy (per subject)")
    ax.set_xlabel("")
    ax.set_title("Group comparison across denoising strategies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
