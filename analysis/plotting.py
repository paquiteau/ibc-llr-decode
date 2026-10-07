"""Shared plotting primitives — one implementation each, reused by subject- and group-level figures."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nilearn.image import load_img, new_img_like
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map
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


def plot_weight_map_comparison(weight_map_paths, bg_img, output_path, title=None):
    """SVM weight maps on the subject's T1w, one panel per strategy, for one condition.

    `weight_map_paths` : dict {strategy: path}, from decode.save_weight_maps()
    output. `bg_img` : the subject's native-space T1w (see glm.find_t1w), same
    space the weight maps were written into.

    Uses nilearn's transparency thresholding (see
    https://nilearn.github.io/stable/auto_examples/01_plotting/plot_transparency.html)
    instead of a hard `threshold=` cutoff: opacity fades from 0 at the median
    nonzero |weight| (pooled across strategies) up to full opacity at the 99th
    percentile, so weak, likely-noisy voxels sink into the anatomy instead of
    popping out at the same color as strong ones. The fade range, color scale,
    and cut coordinates (picked from whichever strategy has the strongest
    voxel, so every panel is sliced through the same anatomical location) are
    all shared across strategies so panels are directly comparable.
    """
    imgs = {strategy: load_img(path) for strategy, path in weight_map_paths.items()}
    vmin_alpha, vmax, cut_coords = _fade_range_and_cut_coords(imgs)

    fig, axes = plt.subplots(len(imgs), 1, figsize=(6, 4 * len(imgs)))
    axes = np.atleast_1d(axes)
    for ax, (strategy, img) in zip(axes, imgs.items()):
        plot_stat_map(
            img,
            bg_img=bg_img,
            cut_coords=cut_coords,
            transparency=img,
            transparency_range=[vmin_alpha, vmax],
            vmax=vmax,
            cmap="cold_hot",
            axes=ax,
            title=strategy,
            colorbar=True,
            draw_cross=False,
        )
    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _fade_range_and_cut_coords(imgs):
    """Shared transparency-thresholding setup for weight/pattern/norm map figures.

    `imgs` : dict {strategy: Nifti1Image}. Opacity fades from 0 at the median
    nonzero |value| (pooled across strategies) up to full opacity at the 99th
    percentile, and cut coordinates are picked from whichever strategy has the
    strongest voxel — both shared across strategies so panels are directly
    comparable (see plot_weight_map_comparison).
    """
    all_abs = np.concatenate([np.abs(img.get_fdata()).ravel() for img in imgs.values()])
    vmin_alpha, vmax = np.percentile(all_abs[all_abs > 0], [50, 99])
    combined = new_img_like(
        next(iter(imgs.values())),
        np.max([np.abs(img.get_fdata()) for img in imgs.values()], axis=0),
    )
    cut_coords = find_xyz_cut_coords(combined)
    return vmin_alpha, vmax, cut_coords


def plot_norm_map_comparison(norm_map_paths, bg_img, output_path, title=None):
    """Per-subject voxel-importance maps (L2 norm across classes) on the T1w,
    one panel per strategy.

    `norm_map_paths` : dict {strategy: path}, from decode.save_norm_map()
    output (see decode.collapse_norm_map — "how much does this voxel matter
    for decoding"). Unlike raw signed weights, norms are one-sided, so this
    uses a sequential colormap with vmin=0 instead of `plot_weight_map_comparison`'s
    diverging one.
    """
    imgs = {strategy: load_img(path) for strategy, path in norm_map_paths.items()}
    vmin_alpha, vmax, cut_coords = _fade_range_and_cut_coords(imgs)

    fig, axes = plt.subplots(len(imgs), 1, figsize=(6, 4 * len(imgs)))
    axes = np.atleast_1d(axes)
    for ax, (strategy, img) in zip(axes, imgs.items()):
        plot_stat_map(
            img,
            bg_img=bg_img,
            cut_coords=cut_coords,
            transparency=img,
            transparency_range=[vmin_alpha, vmax],
            vmin=0,
            vmax=vmax,
            cmap="hot",
            axes=ax,
            title=strategy,
            colorbar=True,
            draw_cross=False,
        )
    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_grid_heatmap(
    df_grid_summary, strategies, output_path, reference_strategy=None
):
    """Mean accuracy over (k, p), one heatmap panel per strategy, plus a
    difference panel when `reference_strategy` is given and there are
    exactly 2 strategies. Sign convention: `gap = other - reference` (matches
    `analysis_results/README.md`'s existing `denoised - fmriprep` convention).

    Magnitude panels share one sequential colormap + color scale (accuracy is
    the same kind of quantity everywhere); the difference panel gets its own
    diverging colormap centered at zero, since a gap can be positive or
    negative.
    """
    show_diff = reference_strategy is not None and len(strategies) == 2
    n_panels = len(strategies) + (1 if show_diff else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    ks = sorted(df_grid_summary["k"].unique())
    ps = sorted(df_grid_summary["p"].unique())
    pivots = {
        strategy: df_grid_summary[df_grid_summary["strategy"] == strategy]
        .pivot(index="k", columns="p", values="mean")
        .reindex(index=ks, columns=ps)
        for strategy in strategies
    }
    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())

    for ax, strategy in zip(axes, strategies):
        sns.heatmap(
            pivots[strategy],
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=True,
        )
        ax.set_title(strategy)
        ax.set_xlabel("p (runs held out)")
        ax.set_ylabel("k (ANOVA features)")

    if show_diff:
        other = next(s for s in strategies if s != reference_strategy)
        gap = pivots[other] - pivots[reference_strategy]
        gap_abs_max = float(np.abs(gap.to_numpy()).max())
        sns.heatmap(
            gap,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            vmin=-gap_abs_max,
            vmax=gap_abs_max,
            center=0,
            ax=axes[-1],
            cbar=True,
        )
        axes[-1].set_title(f"gap ({other} - {reference_strategy})")
        axes[-1].set_xlabel("p (runs held out)")
        axes[-1].set_ylabel("k (ANOVA features)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_grid_lines(df_grid_summary, strategies, output_path):
    """Secondary grid view: accuracy vs k (log scale), one line per p, faceted
    by strategy. p is ordinal (fewer held-out runs = more training data), so
    lines are colored along a single sequential ramp rather than arbitrary
    categorical hues."""
    ps = sorted(df_grid_summary["p"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))

    fig, axes = plt.subplots(
        1, len(strategies), figsize=(5 * len(strategies), 4.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, strategy in zip(axes, strategies):
        sub = df_grid_summary[df_grid_summary["strategy"] == strategy]
        for p, color in zip(ps, colors):
            line = sub[sub["p"] == p].sort_values("k")
            ax.errorbar(
                line["k"],
                line["mean"],
                yerr=line["sem"],
                marker="o",
                color=color,
                label=f"p={p}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("k (ANOVA features)")
        ax.set_title(strategy)
    axes[0].set_ylabel("Mean decoding accuracy")
    axes[-1].legend(title="p (runs held out)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group_comparison(df_long, pairwise_contrasts, output_path, title=None):
    """Group-level box/strip: one point per subject per strategy (mean over folds),
    subjects connected across strategies, significance bars from the fitted LMM contrasts.
    """
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
    ax.set_title(title or "Group comparison across denoising strategies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
