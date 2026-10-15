"""Plotting helpers for pipeline figures."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Reference categorical palette (dataviz skill, light mode), used in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#9a9893"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def library_size_panels(lib: pd.DataFrame, title: str = "Library size per sample") -> plt.Figure:
    """One panel per dataset: library size (millions) by region, box + points."""
    datasets = list(dict.fromkeys(lib["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets) + 1, 3.4), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        sub = lib[lib["dataset"] == name]
        regions = [r for r in dict.fromkeys(sub["region"].astype(str))]
        data = [sub.loc[sub["region"].astype(str) == r, "lib_size"] / 1e6 for r in regions]
        ax.boxplot(data, widths=0.55, showfliers=False,
                   medianprops={"color": SERIES[0], "linewidth": 2},
                   boxprops={"color": TEXT_MUTED}, whiskerprops={"color": TEXT_MUTED},
                   capprops={"color": TEXT_MUTED})
        for i, values in enumerate(data, start=1):
            jitter = [(i + (k % 7 - 3) * 0.04) for k in range(len(values))]
            ax.scatter(jitter, values, s=10, color=SERIES[0], alpha=0.55, linewidths=0)
        ax.set_xticks(range(1, len(regions) + 1), regions, rotation=45, ha="right")
        ax.set_title(name, fontsize=10, color=TEXT)
        _style_axes(ax)
    axes[0][0].set_ylabel("Library size (million reads)", fontsize=9, color=TEXT_MUTED)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def pca_outlier_grid(pca: pd.DataFrame, title: str) -> plt.Figure:
    """Small multiples of PC1 vs PC2 per region; flagged outliers highlighted and labeled."""
    regions = list(dict.fromkeys(pca["region"].astype(str)))
    ncol = min(5, len(regions))
    nrow = math.ceil(len(regions) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.7 * nrow), squeeze=False)
    for ax in axes.flat[len(regions):]:
        ax.set_visible(False)
    for ax, region in zip(axes.flat, regions):
        sub = pca[pca["region"].astype(str) == region]
        normal, flagged = sub[~sub["outlier"]], sub[sub["outlier"]]
        ax.scatter(normal["PC1"], normal["PC2"], s=18, color=NEUTRAL, linewidths=0)
        ax.scatter(flagged["PC1"], flagged["PC2"], s=34, color=SERIES[1],
                   edgecolors="white", linewidths=1.5, zorder=3)
        for _, row in flagged.iterrows():
            ax.annotate(str(row["sample"]), (row["PC1"], row["PC2"]), fontsize=7, color=TEXT,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"{region}  (n={len(sub)}, flagged {len(flagged)})", fontsize=9, color=TEXT)
        ax.set_xlabel(f"PC1 ({sub['PC1_var'].iloc[0]:.0%})", fontsize=8, color=TEXT_MUTED)
        ax.set_ylabel(f"PC2 ({sub['PC2_var'].iloc[0]:.0%})", fontsize=8, color=TEXT_MUTED)
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def cas_vs_age_grid(scores: pd.DataFrame, slopes: pd.DataFrame, region_names: dict[str, str],
                    title: str) -> plt.Figure:
    """Small multiples of CAS vs age per region, ordered by slope; points by sex, fitted line."""
    order = slopes.sort_values("slope", ascending=False)
    ncol = 5
    nrow = math.ceil(len(order) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), squeeze=False, sharex=True)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    sexes = sorted(scores["sex"].unique())
    colors = dict(zip(sexes, SERIES))
    for ax, (_, row) in zip(axes.flat, order.iterrows()):
        sub = scores[scores["region"].astype(str) == row["region"]]
        for sex in sexes:
            s = sub[sub["sex"] == sex]
            ax.scatter(s["age_months"], s["cas"], s=12, color=colors[sex], alpha=0.6, linewidths=0, label=sex)
        means = sub.groupby("age_months")["cas"].mean()
        ax.plot(means.index, means.values, color=TEXT_MUTED, linewidth=1, alpha=0.8)
        xs = pd.Series([sub["age_months"].min(), sub["age_months"].max()])
        ax.plot(xs, row["intercept"] + row["slope"] * xs, color=TEXT, linewidth=2)
        ax.set_title(f"{region_names.get(row['region'], row['region'])}\n"
                     f"slope {row['slope']:.3f}/mo, rho {row['spearman_rho']:.2f}", fontsize=8.5, color=TEXT)
        _style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Age (months)", fontsize=8, color=TEXT_MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("CAS", fontsize=8, color=TEXT_MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    return fig


def slope_ranking(slopes: pd.DataFrame, region_names: dict[str, str], title: str) -> plt.Figure:
    """Horizontal dot-and-CI plot of per-region slopes, fastest on top."""
    order = slopes.sort_values("slope")
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(order) + 1.2))
    y = range(len(order))
    ax.hlines(y, order["slope_ci_low"], order["slope_ci_high"], color=SERIES[0], linewidth=2)
    ax.scatter(order["slope"], y, s=40, color=SERIES[0], edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_yticks(list(y), [region_names.get(r, r) for r in order["region"]], fontsize=8)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_xlabel("CAS slope per month (95% CI)", fontsize=9, color=TEXT_MUTED)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=10, color=TEXT)
    fig.tight_layout()
    return fig


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    # Reference diverging pair: blue <-> red with a neutral gray midpoint.
    return LinearSegmentedColormap.from_list(
        "blue_gray_red", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c9c8", "#e34948", "#a32524"])


def stars(fdr: float) -> str:
    return "***" if fdr < 0.001 else "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""


def effect_heatmap(stats: pd.DataFrame, rows: list[str], cols: list[str], title: str,
                   fdr_col: str = "welch_fdr", vmax: float = 2.0) -> plt.Figure:
    """Region x comparison heatmap of Hedges' g with FDR stars and the g value in each cell."""
    g = stats.pivot(index="region", columns="comparison", values="g").reindex(index=rows, columns=cols)
    fdr = stats.pivot(index="region", columns="comparison", values=fdr_col).reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(1.5 * len(cols) + 2.2, 0.55 * len(rows) + 1.6))
    im = ax.imshow(g.to_numpy(float), cmap=_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = g.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}\n{stars(fdr.iat[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.65 * vmax else TEXT)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0, colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Hedges' g (treatment - control)", fontsize=8, color=TEXT_MUTED)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_MUTED)
    ax.set_title(f"{title}\n* FDR<0.05  ** <0.01  *** <0.001 ({fdr_col})", fontsize=10, color=TEXT)
    fig.tight_layout()
    return fig


def group_boxplots(scores: pd.DataFrame, regions: list[str], groups: list[str], color_by: dict[str, str],
                   title: str, value: str = "cas") -> plt.Figure:
    """One panel per region; box + points per group, colored by a two-level factor (e.g. drug)."""
    ncol = min(3, len(regions))
    nrow = math.ceil(len(regions) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat[len(regions):]:
        ax.set_visible(False)
    levels = list(dict.fromkeys(color_by.values()))
    palette = {lvl: c for lvl, c in zip(levels, [NEUTRAL, SERIES[1], SERIES[0]])}
    for ax, region in zip(axes.flat, regions):
        sub = scores[scores["region"].astype(str) == region]
        data = [sub.loc[sub["group"].astype(str) == g, value] for g in groups]
        ax.boxplot(data, widths=0.55, showfliers=False, medianprops={"color": TEXT, "linewidth": 1.5},
                   boxprops={"color": TEXT_MUTED}, whiskerprops={"color": TEXT_MUTED}, capprops={"color": TEXT_MUTED})
        for i, (grp, vals) in enumerate(zip(groups, data), start=1):
            jitter = [i + (k % 5 - 2) * 0.06 for k in range(len(vals))]
            ax.scatter(jitter, vals, s=16, color=palette[color_by[grp]], linewidths=0, alpha=0.85, zorder=3)
        ax.set_xticks(range(1, len(groups) + 1), groups, fontsize=8)
        ax.set_title(region, fontsize=10, color=TEXT)
        _style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("CAS", fontsize=9, color=TEXT_MUTED)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", color=palette[lvl], label=lvl) for lvl in levels]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    return fig


def gene_lfc_heatmap(lfc: pd.DataFrame, gene_order: list[str], col_order: list[str], gene_group: dict[str, str],
                     gene_direction: dict[str, str], title: str, vmax: float = 1.0) -> plt.Figure:
    """Genes x (region | comparison) log2FC heatmap; rows grouped by function, aging direction marked."""
    mat = lfc.pivot(index="gene", columns="column", values="log2FoldChange").reindex(index=gene_order, columns=col_order)
    fig, ax = plt.subplots(figsize=(0.32 * len(col_order) + 4, 0.16 * len(gene_order) + 2))
    im = ax.imshow(mat.to_numpy(float), cmap=_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    labels = [f"{g} {'↑' if gene_direction[g] == 'up' else '↓'}" for g in gene_order]
    ax.set_yticks(range(len(gene_order)), labels, fontsize=5.5)
    ax.set_xticks(range(len(col_order)), col_order, fontsize=6.5, rotation=90)
    groups = [gene_group[g] for g in gene_order]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=2.5)
    starts = {}
    for i, grp in enumerate(groups):
        starts.setdefault(grp, [i, i])[1] = i
    for grp, (a, b) in starts.items():
        ax.text(len(col_order) - 0.3, (a + b) / 2, f" {grp}", va="center", ha="left", fontsize=7, color=TEXT)
    regions = [c.split(" | ")[0] for c in col_order]
    for j in range(1, len(regions)):
        if regions[j] != regions[j - 1]:
            ax.axvline(j - 0.5, color="white", linewidth=2.5)
    ax.tick_params(length=0, colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.16)
    cbar.set_label(f"log2 fold change (clipped at ±{vmax})", fontsize=7, color=TEXT_MUTED)
    cbar.ax.tick_params(labelsize=6, colors=TEXT_MUTED)
    ax.set_title(f"{title}\n↑/↓ = direction with age in Hahn 2023", fontsize=10, color=TEXT)
    fig.tight_layout()
    return fig
