from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#183153"
BLUE = "#4D73E6"
TEAL = "#159A9C"
GOLD = "#E9A23B"
MUTED = "#667085"
LIGHT = "#F5F7FB"
BORDER = "#D7DEEA"


def rounded(ax, x, y, width, height, color="white", edge=BORDER, radius=0.16, linewidth=1.3):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=18,
            linewidth=1.8, color="#98A2B3",
        )
    )


def chip(ax, x, y, width, text, color=LIGHT, text_color=NAVY, fontsize=10):
    rounded(ax, x, y, width, 0.38, color=color, edge=color, radius=0.1)
    ax.text(x + width / 2, y + 0.19, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def overview(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.7)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0.65, 6.18, "From denoising traces to verified concepts",
        fontsize=22, fontweight="bold", color=NAVY, va="center",
    )
    ax.text(
        0.67, 5.74, "Candidate construction and evidence acceptance are measured separately.",
        fontsize=11.5, color=MUTED, va="center",
    )

    xs = [0.62, 4.05, 7.48, 10.91]
    labels = ["01  SUPPORT", "02  DENOISE", "03  CONSTRUCT", "04  VERIFY"]
    titles = ["Support\nexperiences", "Revisable\ndenoising", "Construct\ncandidates", "Evidence\nacceptance"]
    accents = [GOLD, BLUE, TEAL, NAVY]
    for x, label, title, accent in zip(xs, labels, titles, accents):
        rounded(ax, x, 1.72, 2.48, 3.52, color="white")
        rounded(ax, x, 5.08, 2.48, 0.16, color=accent, edge=accent, radius=0.06, linewidth=0)
        ax.text(x + 0.22, 4.79, label, fontsize=9.2, color=accent, fontweight="bold")
        ax.text(
            x + 0.22, 4.25, title, fontsize=15, color=NAVY,
            fontweight="bold", va="center", linespacing=1.18,
        )

    chip(ax, 0.88, 3.20, 1.96, "object A   →   +", color=LIGHT, fontsize=9.5)
    chip(ax, 0.88, 2.68, 1.96, "object B   →   −", color=LIGHT, fontsize=9.5)
    ax.text(0.89, 2.08, "a few labeled examples", fontsize=9.5, color=MUTED)

    for index, text in enumerate(("[M]   [M]   [M]", "red   [M]   [M]", "red   square   [M]")):
        chip(ax, 4.31, 3.48 - index * 0.51, 1.96, text, color=LIGHT, fontsize=9.3)
    ax.text(5.29, 2.08, "samples + remasking", fontsize=9.2, color=MUTED, ha="center")

    chip(ax, 7.74, 3.48, 1.96, "red", color=LIGHT, fontsize=9.5)
    chip(ax, 7.74, 2.97, 1.96, "red ∧ square", color="#EEF2FF", fontsize=9.5)
    chip(ax, 7.74, 2.46, 1.96, "red ∧ circle", color=LIGHT, fontsize=9.5)
    ax.text(8.72, 2.08, "add • remove • merge", fontsize=9.0, color=MUTED, ha="center")

    chip(ax, 11.17, 3.48, 1.96, "held-out experiences", color=LIGHT, fontsize=9.2)
    chip(ax, 11.17, 2.97, 1.96, "score • select • abstain", color=LIGHT, fontsize=9.2)
    rounded(ax, 11.17, 2.04, 1.96, 0.58, color="#E8F7F4", edge="#BFE6DE", radius=0.10)
    ax.text(12.15, 2.33, "executable concept", fontsize=9.7, color=TEAL, fontweight="bold", ha="center", va="center")

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + 2.61, 3.45), (right - 0.13, 3.45))

    rounded(ax, 0.64, 0.43, 12.73, 0.80, color=NAVY, edge=NAVY, radius=0.13)
    ax.text(3.70, 0.96, "CANDIDATE RECALL", color="#BFD0F7", fontsize=10, fontweight="bold", ha="center")
    ax.text(3.70, 0.67, "Did the true concept appear?", color="white", fontsize=11, ha="center")
    ax.text(7.00, 0.83, "≠", color=GOLD, fontsize=22, fontweight="bold", ha="center", va="center")
    ax.text(10.25, 0.96, "EXACT RECOVERY", color="#BFE6DE", fontsize=10, fontweight="bold", ha="center")
    ax.text(10.25, 0.67, "Was the true concept selected?", color="white", fontsize=11, ha="center")

    svg_path = output / "method_overview.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(output / "method_overview.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_metrics(methods: pd.DataFrame, ablations: pd.DataFrame, task: str, split: str, method: str):
    source = ablations if method in {"greedy_search", "beam_search"} else methods
    row = source[
        (source["task"] == task) & (source["split"] == split) & (source["method"] == method)
    ]
    if row.empty:
        raise ValueError((task, split, method))
    row = row.iloc[0]
    return float(row["concept_accuracy_mean"]), float(row["concept_accuracy_std"])


def results_figure(methods: pd.DataFrame, ablations: pd.DataFrame, output: Path) -> None:
    names = ["evidence_guided", "random_search_matched", "random_search", "learned_search", "greedy_search", "beam_search", "symbolic"]
    labels = ["Trace\nselection", "Random\nmatched", "Random\n96", "Learned\nsearch", "Greedy", "Beam", "Symbolic"]
    colors = ["#91A7D9", "#B8C0CC", "#98A2B3", BLUE, TEAL, "#54B8B2", NAVY]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.patch.set_facecolor("white")
    for axis, task, title, subtitle in (
        (axes[0], "dcoder", "Open-dCoder 0.5B", "Unseen three-attribute concepts • 3 seeds"),
        (axes[1], "large", "4,064-concept space", "Unseen three-attribute concepts • 1 seed"),
    ):
        values = []
        errors = []
        for method in names:
            value, error = selected_metrics(methods, ablations, task, "compositional", method)
            values.append(value)
            errors.append(0.0 if np.isnan(error) else error)
        x = np.arange(len(names))
        has_errors = task == "dcoder"
        bars = axis.bar(
            x, values, yerr=errors if has_errors else None, capsize=4 if has_errors else 0,
            color=colors, width=0.74, zorder=3,
        )
        axis.set_ylim(0, 1.06)
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylabel("Exact concept recovery", fontsize=11)
        axis.set_title(title, fontsize=16, fontweight="bold", color=NAVY, pad=22)
        axis.text(0.5, 1.015, subtitle, transform=axis.transAxes, ha="center", fontsize=10, color=MUTED)
        axis.grid(axis="y", alpha=0.20, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value, error in zip(bars, values, errors):
            offset = error if has_errors else 0.0
            axis.text(
                bar.get_x() + bar.get_width() / 2, min(value + offset + 0.035, 1.025), f"{value:.3f}",
                ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold",
            )
    fig.suptitle(
        "Construction helps — but classical and symbolic controls remain strong",
        fontsize=19, fontweight="bold", color=NAVY, y=1.015,
    )
    fig.text(
        0.5, 0.012,
        "Exact concept recovery, not neural query accuracy. Error bars are seed standard deviations where available. "
        "Search budgets and compute are reported separately in the paper.",
        ha="center", fontsize=9.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.98), w_pad=3.0)
    fig.savefig(output / "main_results.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    source = root / "outputs" / "final_statistics"
    if not (source / "method_metrics.csv").exists():
        source = root / "docs" / "results"
    methods = pd.read_csv(source / "method_metrics.csv")
    ablations = pd.read_csv(source / "ablation_metrics.csv")
    overview(output)
    results_figure(methods, ablations, output)
    print(f"Wrote README figures to {output}")


if __name__ == "__main__":
    main()
