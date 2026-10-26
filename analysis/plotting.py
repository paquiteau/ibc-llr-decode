"""Figures for the strategy comparison, styled for print.

Sizes follow the ISBI template geometry (spconf.sty: 86 mm column, 178 mm
text width). Fonts are embedded as TrueType (pdf.fonttype 42): IEEE PDF
eXpress rejects the Type 3 fonts matplotlib embeds by default.
"""

import functools
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from nilearn.image import load_img, new_img_like
from nilearn.plotting import find_xyz_cut_coords, plot_stat_map

from _constants import CONDITION_LABELS, STRATEGY_LABELS
from analysis.stats import mean_ci

COLUMN_WIDTH = 86 / 25.4
TEXT_WIDTH = 178 / 25.4

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
PAIR_LINE = "#c3c2b7"
SURFACE = "#ffffff"
# Reference strategy in neutral ink, the others in categorical slots. Checked
# as a set (OKLab dE x100): >= 15 under normal vision and >= 8 under simulated
# protan/deutan/tritan vision for every pair.
REFERENCE_COLOR = INK_SECONDARY
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_div",
    ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f6b8b7", "#e34948", "#9e2a2a"],
)

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "TeX Gyre Heros", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _paper_style(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(PAPER_RC):
            return func(*args, **kwargs)

    return wrapper


def strategy_label(strategy, width=None):
    label = STRATEGY_LABELS.get(strategy, strategy)
    return textwrap.fill(label, width) if width else label


def condition_label(condition):
    return CONDITION_LABELS.get(condition, condition)


def strategy_colors(strategies, reference):
    """Fixed color per strategy: reference in neutral ink, others in slot order.

    Past the validated slots, every non-reference strategy shares slot 1 and
    is told apart by its axis position only.
    """
    others = [s for s in strategies if s != reference]
    colors = {reference: REFERENCE_COLOR}
    if len(others) <= len(SERIES_COLORS):
        colors.update(zip(others, SERIES_COLORS))
    else:
        colors.update(dict.fromkeys(others, SERIES_COLORS[0]))
    return colors


def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def stars_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _savefig(fig, output_path, png=False):
    fig.savefig(output_path)
    if png:
        fig.savefig(Path(output_path).with_suffix(".png"))
    plt.close(fig)


def _panel_label(ax, letter):
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-30, 4),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _swarm_offsets(values, width=0.07, min_gap=None):
    """Horizontal offsets so that points with close values don't overlap (1D swarm)."""
    values = np.asarray(values, dtype=float)
    min_gap = min_gap or 0.03 * (np.ptp(values) or 1.0)
    offsets = np.zeros_like(values)
    placed = []
    for i in np.argsort(values):
        for step in range(len(values)):
            candidate = (step + 1) // 2 * width * (1 if step % 2 else -1)
            if all(
                abs(values[i] - values[j]) >= min_gap
                or abs(candidate - offsets[j]) >= width
                for j in placed
            ):
                break
        offsets[i] = candidate
        placed.append(i)
    return offsets


def _paired_panel(ax, table, strategies, colors, show_ci=True):
    """One line per row of `table` (subjects or folds) across strategies, plus
    each strategy's mean (+ 95% CI) drawn beside its points."""
    x = np.arange(len(strategies))
    for _, row in table[strategies].iterrows():
        ax.plot(x, row.to_numpy(), color=PAIR_LINE, linewidth=0.6, zorder=1)
    for i, strategy in enumerate(strategies):
        ax.scatter(
            np.full(len(table), i),
            table[strategy],
            s=12,
            color=colors[strategy],
            edgecolor=SURFACE,
            linewidth=0.5,
            zorder=2,
        )
        mean, low, high = mean_ci(table[strategy])
        x_mean = i + (-0.2 if i == 0 else 0.2)
        ax.errorbar(
            x_mean,
            mean,
            yerr=[[mean - low], [high - mean]] if show_ci else None,
            fmt="o",
            markersize=4.5,
            color=colors[strategy],
            markeredgecolor=SURFACE,
            markeredgewidth=0.6,
            elinewidth=1.2,
            capsize=0,
            zorder=3,
        )
    ax.set_xticks(x, [strategy_label(s, 14) for s in strategies])
    ax.set_xlim(-0.5, len(strategies) - 0.5)
    ax.tick_params(axis="x", length=0)


def _chance_line(ax, x_text, chance):
    ax.axhline(100 * chance, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(
        x_text,
        100 * chance,
        "chance",
        color=INK_MUTED,
        fontsize=6.5,
        ha="right",
        va="bottom",
    )


@_paper_style
def plot_group_accuracy(means, contrasts, reference, output_path, chance):
    """Group figure: (a) per-subject accuracy across strategies, lines joining a
    subject's values, group mean +/- 95% CI beside; (b) paired difference to
    the reference per subject, with the group mean +/- 95% CI and the
    sign-flip p-value (Holm-corrected when there are several comparisons).

    `means` : subject x strategy accuracy (`stats.subject_means`), reference
    column first. `contrasts` : `stats.group_contrasts` output.
    """
    strategies = list(means.columns)
    others = [s for s in strategies if s != reference]
    colors = strategy_colors(strategies, reference)
    acc = 100 * means

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(COLUMN_WIDTH, 2.25),
        gridspec_kw=dict(width_ratios=[len(strategies), len(others) + 0.4]),
        layout="constrained",
    )
    _paired_panel(ax_a, acc, strategies, colors)
    _chance_line(ax_a, len(strategies) - 0.5, chance)
    ax_a.set_ylim(10, np.ceil(acc.to_numpy().max() / 5) * 5 + 2)
    ax_a.set_ylabel("Decoding accuracy (%)")
    _panel_label(ax_a, "a")

    p_col = "p_holm" if len(others) > 1 else "p_perm"
    data_low, data_high = 0.0, 0.0
    for j, other in enumerate(others):
        diff = acc[other] - acc[reference]
        ax_b.scatter(
            j - 0.1 + _swarm_offsets(diff),
            diff,
            s=12,
            color=colors[other],
            edgecolor=SURFACE,
            linewidth=0.5,
            zorder=2,
        )
        row = contrasts.set_index("strategy_b").loc[other]
        mean, low, high = 100 * row["diff"], 100 * row["ci_low"], 100 * row["ci_high"]
        ax_b.errorbar(
            j + 0.22,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=4.5,
            color=colors[other],
            markeredgecolor=SURFACE,
            markeredgewidth=0.6,
            elinewidth=1.2,
            capsize=0,
            zorder=3,
        )
        data_low = min(data_low, diff.min(), low)
        data_high = max(data_high, diff.max(), high)
        ax_b.annotate(
            f"{mean:+.1f} pp\n{format_p(row[p_col])}",
            xy=(j + 0.05, max(diff.max(), high)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax_b.axhline(0, color=INK_MUTED, linewidth=0.6, zorder=1)
    ax_b.set_xticks(range(len(others)), [strategy_label(s, 14) for s in others])
    ax_b.tick_params(axis="x", length=0)
    ax_b.set_xlim(-0.6, len(others) - 0.4)
    span = data_high - data_low
    ax_b.set_ylim(data_low - 0.08 * span, data_high + 0.3 * span)
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax_b.set_ylabel(f"\u0394 accuracy vs. {strategy_label(reference)} (pp)")
    _panel_label(ax_b, "b")

    _savefig(fig, output_path, png=True)


@_paper_style
def plot_training_size(df_grid, contrasts, reference, output_path):
    """Accuracy as training data shrinks (more runs held out per CV fold):
    (a) group mean +/- 95% CI per strategy; (b) per-subject paired difference
    to the reference (thin lines) with its group mean +/- 95% CI, starred by
    the Holm-corrected sign-flip p-value.

    `df_grid` : `stats.build_grid_long_dataframe` at a single k.
    `contrasts` : `stats.training_size_contrasts` output.
    """
    others = list(dict.fromkeys(contrasts["strategy_b"]))
    strategies = [reference, *others]
    colors = strategy_colors(strategies, reference)
    acc = _training_accuracy(df_grid)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 1.95), layout="constrained"
    )
    _accuracy_vs_training_panel(ax_a, acc, strategies, colors)
    fig.legend(
        *ax_a.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=len(strategies),
        handlelength=1.5,
    )
    _panel_label(ax_a, "a")

    _gain_vs_training_panel(ax_b, acc, contrasts, reference, others, colors)
    _panel_label(ax_b, "b")

    _savefig(fig, output_path, png=True)


def _training_accuracy(df_grid):
    """(subject, n_train_runs) x strategy accuracy table, in percent."""
    return 100 * df_grid.pivot_table(
        index=["subject", "n_train_runs"],
        columns="strategy",
        values="accuracy",
        aggfunc="mean",
    )


def _accuracy_vs_training_panel(ax, acc, strategies, colors, contrasts=None):
    """Group mean +/- 95% CI accuracy per strategy vs. number of training runs.
    With `contrasts` (`stats.training_size_contrasts`), each non-reference
    strategy is starred by the Holm-corrected sign-flip p-value of its paired
    difference to the reference."""
    n_train = np.array(sorted(acc.index.unique("n_train_runs")))
    dodge = np.linspace(-0.06, 0.06, len(strategies))
    for offset, strategy in zip(dodge, strategies):
        stats_ = np.array(
            [mean_ci(acc[strategy].xs(n, level="n_train_runs")) for n in n_train]
        )
        ax.errorbar(
            n_train + offset,
            stats_[:, 0],
            yerr=[stats_[:, 0] - stats_[:, 1], stats_[:, 2] - stats_[:, 0]],
            color=colors[strategy],
            marker="o",
            markersize=3.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            linewidth=1.2,
            elinewidth=0.8,
            capsize=0,
            label=strategy_label(strategy),
        )
        if contrasts is not None and strategy in set(contrasts["strategy_b"]):
            rows = contrasts[contrasts["strategy_b"] == strategy]
            p = rows.set_index("n_train_runs").loc[n_train, "p_holm"]
            for x, y, p_ in zip(n_train + offset, stats_[:, 2], p):
                ax.text(x, y, stars_from_p(p_), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(n_train)
    ax.set_xlabel("Training runs per fold")
    ax.set_ylabel("Decoding accuracy (%)")


def _gain_vs_training_panel(ax, acc, contrasts, reference, others, colors):
    """Per-subject paired gain vs. number of training runs (thin lines), with
    the group mean +/- 95% CI starred by the Holm-corrected sign-flip p-value."""
    n_train = np.array(sorted(acc.index.unique("n_train_runs")))
    for other in others:
        diff = (acc[other] - acc[reference]).unstack("n_train_runs")[n_train]
        for _, row in diff.iterrows():
            ax.plot(n_train, row.to_numpy(), color=PAIR_LINE, linewidth=0.5, zorder=1)
        rows = contrasts[contrasts["strategy_b"] == other].set_index("n_train_runs")
        rows = rows.loc[n_train]
        mean, low, high = (
            100 * rows[c].to_numpy() for c in ("diff", "ci_low", "ci_high")
        )
        ax.errorbar(
            n_train,
            mean,
            yerr=[mean - low, high - mean],
            color=colors[other],
            marker="o",
            markersize=3.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            linewidth=1.2,
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )
        for x, y, p in zip(n_train, high, rows["p_holm"]):
            ax.text(x, y, stars_from_p(p), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color=INK_MUTED, linewidth=0.6, zorder=1)
    ax.set_xticks(n_train)
    ax.set_xlabel("Training runs per fold")
    ax.set_ylabel("\u0394 accuracy (pp)")


def _per_class_panel(ax, confusions, labels, per_class, reference, others, colors):
    """Per-class recall gain: subjects (gray dots) and group mean +/- 95% CI,
    starred by the Holm-corrected sign-flip p-value. `confusions` holds the
    per-subject matrices of `stats.subject_confusion_matrices`."""
    y = np.arange(len(labels))
    dodge = np.linspace(-0.15, 0.15, len(others)) if len(others) > 1 else [0.0]
    for offset, other in zip(dodge, others):
        rows = (
            per_class[per_class["strategy_b"] == other].set_index("label").loc[labels]
        )
        for i, label in enumerate(labels):
            subject_diff = 100 * (
                confusions[other][:, i, i] - confusions[reference][:, i, i]
            )
            ax.scatter(
                subject_diff,
                np.full(subject_diff.size, i + offset),
                s=5,
                color=PAIR_LINE,
                linewidth=0,
                zorder=1,
            )
        mean, low, high = (
            100 * rows[c].to_numpy() for c in ("diff", "ci_low", "ci_high")
        )
        ax.errorbar(
            mean,
            y + offset,
            xerr=[mean - low, high - mean],
            fmt="o",
            markersize=3.5,
            color=colors[other],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            elinewidth=1.0,
            capsize=0,
            zorder=3,
        )
        for yi, x, p in zip(y + offset, high, rows["p_holm"]):
            ax.text(x + 0.5, yi, stars_from_p(p), ha="left", va="center", fontsize=7)
    ax.axvline(0, color=INK_MUTED, linewidth=0.6, zorder=0)
    ax.set_yticks(y, [condition_label(label) for label in labels])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("\u0394 recall (pp)")


@_paper_style
def plot_decoding_summary(
    means,
    df_grid,
    size_contrasts,
    confusions,
    labels,
    per_class,
    reference,
    output_path,
    chance,
):
    """Full-width summary figure: (a) per-subject leave-one-run-out accuracy,
    (b) accuracy vs. number of training runs, starred by the paired test,
    (c) per-class recall gain. One legend for the strategies, above the panels.

    `means` : `stats.subject_means`, reference first. `df_grid`,
    `size_contrasts` : as for `plot_training_size`. `confusions`, `labels`,
    `per_class` : per-subject confusion matrices and `stats.per_class_contrasts`.
    """
    strategies = list(means.columns)
    others = [s for s in strategies if s != reference]
    colors = strategy_colors(strategies, reference)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(TEXT_WIDTH, 2.0),
        gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.1]),
        layout="constrained",
    )
    acc = 100 * means
    _paired_panel(ax_a, acc, strategies, colors)
    _chance_line(ax_a, len(strategies) - 0.5, chance)
    ax_a.set_ylim(10, np.ceil(acc.to_numpy().max() / 5) * 5 + 2)
    ax_a.set_ylabel("Decoding accuracy (%)")
    _accuracy_vs_training_panel(
        ax_b, _training_accuracy(df_grid), strategies, colors, size_contrasts
    )
    _per_class_panel(ax_c, confusions, labels, per_class, reference, others, colors)
    fig.legend(
        *ax_b.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=len(strategies),
        handlelength=1.5,
    )
    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        _panel_label(ax, letter)
    _savefig(fig, output_path, png=True)


def _annotate_cells(ax, matrix, fmt, threshold, suffixes=None):
    for (i, j), value in np.ndenumerate(matrix):
        text = fmt(value) + (suffixes[i][j] if suffixes is not None else "")
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=6,
            color=SURFACE if abs(float(value)) > threshold else INK,
        )


@_paper_style
def plot_confusion_matrices(
    confusions, labels, reference, output_path, per_class=None, png=False
):
    """Row-normalized confusion matrices (diagonal = per-class recall), one row
    per non-reference strategy: reference, that strategy, and their difference.

    `confusions` : {strategy: (n_classes, n_classes)} matrix, e.g. the mean
    over subjects of `stats.subject_confusion_matrices`, reference first.
    `per_class` : optional `stats.per_class_contrasts` output; diagonal cells
    of the difference are starred by their Holm-corrected p-value.
    """
    confusions = {s: 100 * m for s, m in confusions.items()}
    others = [s for s in confusions if s != reference]
    ticks = [condition_label(label) for label in labels]
    vmax = max(m.max() for m in confusions.values())
    dmax = max(np.abs(confusions[o] - confusions[reference]).max() for o in others)

    fig, axes = plt.subplots(
        len(others),
        3,
        figsize=(TEXT_WIDTH, 2.35 * len(others)),
        squeeze=False,
        layout="constrained",
    )
    for row, other in zip(axes, others):
        for ax, strategy in zip(row[:2], (reference, other)):
            im = ax.imshow(
                confusions[strategy], cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax
            )
            _annotate_cells(ax, confusions[strategy], lambda v: f"{v:.0f}", 0.6 * vmax)
            ax.set_title(strategy_label(strategy))
        diff = confusions[other] - confusions[reference]
        stars = [[""] * len(labels) for _ in labels]
        if per_class is not None:
            p = per_class[per_class["strategy_b"] == other].set_index("label")["p_holm"]
            for i, label in enumerate(labels):
                stars[i][i] = stars_from_p(p[label])
        im_diff = row[2].imshow(diff, cmap=DIVERGING_CMAP, vmin=-dmax, vmax=dmax)
        _annotate_cells(
            row[2],
            diff,
            lambda v: f"{round(v):+d}".replace("-", "\u2212") if round(v) else "0",
            0.6 * dmax,
            stars,
        )
        row[2].set_title("Difference")
        for ax in row:
            ax.set_xticks(range(len(labels)), ticks, rotation=40, ha="right")
            ax.set_yticks(range(len(labels)), ticks if ax is row[0] else [])
            ax.tick_params(length=0)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("Predicted")
        row[0].set_ylabel("True")
        fig.colorbar(
            im,
            ax=row[:2],
            shrink=0.8,
            label="Trials of the true class (%)",
        )
        cbar = fig.colorbar(im_diff, ax=row[2], shrink=0.8, label="\u0394 (pp)")
        cbar.locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True)
        cbar.update_ticks()

    _savefig(fig, output_path, png=png)


@_paper_style
def plot_subject_folds(
    df_subject_long, test_rows, reference, subject, output_path, chance
):
    """Per-subject accuracy per CV fold (one line per held-out run) across
    strategies, titled with the single-subject sign-flip test vs the reference
    (`stats.subject_block_tests`)."""
    strategies = [reference] + [
        s for s in df_subject_long["strategy"].unique() if s != reference
    ]
    acc = 100 * df_subject_long.pivot(
        index="fold", columns="strategy", values="accuracy"
    )
    fig, ax = plt.subplots(
        figsize=(0.75 * len(strategies) + 1.0, 2.1), layout="constrained"
    )
    _paired_panel(
        ax, acc, strategies, strategy_colors(strategies, reference), show_ci=False
    )
    _chance_line(ax, len(strategies) - 0.5, chance)
    ax.set_ylim(10, max(70, acc.to_numpy().max() + 5))
    ax.set_ylabel("Accuracy per held-out run (%)")
    tests = "; ".join(
        f"{format_p(r.p_signflip)}"
        + (f" ({strategy_label(r.strategy_b)})" if len(test_rows) > 1 else "")
        for r in test_rows.itertuples()
    )
    ax.set_title(f"sub-{subject}  sign-flip {tests}", fontsize=7)
    _savefig(fig, output_path)


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


def _plot_map_comparison(imgs, bg_img, output_path, title, **stat_map_kwargs):
    vmin_alpha, vmax, cut_coords = _fade_range_and_cut_coords(imgs)
    fig, axes = plt.subplots(len(imgs), 1, figsize=(TEXT_WIDTH, 2.0 * len(imgs)))
    for ax, (strategy, img) in zip(np.atleast_1d(axes), imgs.items()):
        display = plot_stat_map(
            img,
            bg_img=bg_img,
            cut_coords=cut_coords,
            transparency=img,
            transparency_range=[vmin_alpha, vmax],
            vmax=vmax,
            axes=ax,
            colorbar=True,
            draw_cross=False,
            annotate=False,
            **stat_map_kwargs,
        )
        display.title(strategy_label(strategy), size=8, color="white", bgcolor="black")
    if title:
        fig.suptitle(title)
    _savefig(fig, output_path)


@_paper_style
def plot_weight_map_comparison(weight_map_paths, bg_img, output_path, title=None):
    """SVM weight maps on the subject's T1w, one panel per strategy, for one condition.

    `weight_map_paths` : dict {strategy: path or image}, from
    decode.save_weight_maps() output. `bg_img` : the subject's native-space
    T1w (see glm.find_t1w), same space the weight maps were written into.

    Uses nilearn's transparency thresholding (see
    https://nilearn.github.io/stable/auto_examples/01_plotting/plot_transparency.html)
    instead of a hard `threshold=` cutoff: opacity fades from 0 at the median
    nonzero |weight| (pooled across strategies) up to full opacity at the 99th
    percentile, so weak, likely-noisy voxels sink into the anatomy instead of
    popping out at the same color as strong ones. The fade range, color scale,
    and cut coordinates are shared across strategies so panels are directly
    comparable.
    """
    imgs = {strategy: load_img(path) for strategy, path in weight_map_paths.items()}
    _plot_map_comparison(imgs, bg_img, output_path, title, cmap="cold_hot")


@_paper_style
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
    _plot_map_comparison(imgs, bg_img, output_path, title, cmap="hot", vmin=0)


@_paper_style
def plot_grid_heatmap(
    df_grid_summary, strategies, output_path, reference_strategy=None
):
    """Mean accuracy over (k, p), one heatmap panel per strategy, plus a
    difference panel when `reference_strategy` is given and there are
    exactly 2 strategies. Sign convention: `gap = other - reference`.

    Magnitude panels share one sequential colormap + color scale (accuracy is
    the same kind of quantity everywhere); the difference panel gets its own
    diverging colormap centered at zero, since a gap can be positive or
    negative.
    """
    show_diff = reference_strategy is not None and len(strategies) == 2
    n_panels = len(strategies) + (1 if show_diff else 0)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(TEXT_WIDTH / 3 * n_panels, 2.4),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]

    ks = sorted(df_grid_summary["k"].unique())
    ps = sorted(df_grid_summary["p"].unique())
    pivots = {
        strategy: 100
        * df_grid_summary[df_grid_summary["strategy"] == strategy]
        .pivot(index="k", columns="p", values="mean")
        .reindex(index=ks, columns=ps)
        for strategy in strategies
    }
    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())
    heatmap_kwargs = dict(annot=True, fmt=".1f", annot_kws=dict(fontsize=6), cbar=True)

    for ax, strategy in zip(axes, strategies):
        sns.heatmap(
            pivots[strategy],
            cmap=SEQUENTIAL_CMAP,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **heatmap_kwargs,
        )
        ax.set_title(f"{strategy_label(strategy)} (%)")

    if show_diff:
        other = next(s for s in strategies if s != reference_strategy)
        gap = pivots[other] - pivots[reference_strategy]
        gap_abs_max = float(np.abs(gap.to_numpy()).max())
        sns.heatmap(
            gap,
            cmap=DIVERGING_CMAP,
            vmin=-gap_abs_max,
            vmax=gap_abs_max,
            center=0,
            ax=axes[-1],
            **heatmap_kwargs,
        )
        axes[-1].set_title("Difference (pp)")
    for ax in axes:
        ax.set_xlabel("Runs held out (p)")
        ax.set_ylabel("ANOVA features (k)")

    _savefig(fig, output_path)


@_paper_style
def plot_grid_lines(df_grid_summary, strategies, output_path):
    """Secondary grid view: accuracy vs k (log scale), one line per p, faceted
    by strategy. p is ordinal (fewer held-out runs = more training data), so
    lines are colored along a single sequential ramp rather than arbitrary
    categorical hues."""
    ps = sorted(df_grid_summary["p"].unique())
    colors = SEQUENTIAL_CMAP(np.linspace(0.4, 1, len(ps)))

    fig, axes = plt.subplots(
        1,
        len(strategies),
        figsize=(TEXT_WIDTH / 3 * len(strategies), 2.2),
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for ax, strategy in zip(axes, strategies):
        sub = df_grid_summary[df_grid_summary["strategy"] == strategy]
        for p, color in zip(ps, colors):
            line = sub[sub["p"] == p].sort_values("k")
            ax.errorbar(
                line["k"],
                100 * line["mean"],
                yerr=100 * line["sem"],
                marker="o",
                markersize=3,
                linewidth=1.0,
                elinewidth=0.8,
                capsize=0,
                color=color,
                label=f"p = {p}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("ANOVA features (k)")
        ax.set_title(strategy_label(strategy))
    axes[0].set_ylabel("Decoding accuracy (%)")
    axes[-1].legend(title="Runs held out")
    _savefig(fig, output_path)
