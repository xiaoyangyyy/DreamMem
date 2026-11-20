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
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.65, 6.03, "DreamMem", fontsize=25, fontweight="bold", color=NAVY, va="center")
    ax.text(
        4.05, 6.03, "Evidence-guided constructive memory in masked diffusion LMs",
        fontsize=15, color=NAVY, va="center",
    )
    ax.text(
        0.67, 5.57,
        "Turn revisable denoising hypotheses into compact executable concepts — not merely more generated text.",
        fontsize=11.5, color=MUTED, va="center",
    )

    boxes = [(0.62, 1.75), (5.16, 1.75), (9.70, 1.75)]
    for x, y in boxes:
        rounded(ax, x, y, 3.68, 3.45, color="white")
    for number, (x, _), title in zip(
        (1, 2, 3), boxes,
        ("Revisable denoising", "Candidate construction", "Evidence acceptance"),
    ):
        ax.add_patch(plt.Circle((x + 0.34, 4.79), 0.20, color=BLUE))
        ax.text(x + 0.34, 4.79, str(number), color="white", fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(x + 0.65, 4.79, title, color=NAVY, fontsize=12.2, fontweight="bold", va="center")

    ax.text(0.91, 4.30, "labeled experiences + masked concept slots", fontsize=9.5, color=MUTED)
    for index, (text, color) in enumerate(
        (("red    [M]    [M]", LIGHT), ("red    square    [M]", "#EEF2FF"), ("red    square    object", "#E8F7F4"))
    ):
        chip(ax, 0.93, 3.72 - index * 0.53, 3.04, text, color=color, fontsize=9.5)
    ax.text(0.94, 2.00, "multiple samples • remasking • intermediate states", fontsize=9.2, color=MUTED)

    ax.text(5.45, 4.30, "observed states + symbolic neighborhoods", fontsize=9.5, color=MUTED)
    chip(ax, 5.47, 3.72, 1.29, "red", color=LIGHT, fontsize=9.5)
    chip(ax, 6.89, 3.72, 1.60, "red ∧ square", color="#EEF2FF", fontsize=9.5)
    chip(ax, 5.47, 3.19, 1.70, "red ∧ circle", color=LIGHT, fontsize=9.5)
    chip(ax, 7.30, 3.19, 1.19, "blue ∧ …", color=LIGHT, fontsize=9.5)
    ax.text(5.48, 2.65, "add  •  remove  •  substitute", fontsize=10, color=TEAL, fontweight="bold")
    ax.text(5.48, 2.28, "merge  •  closure  •  learned proposals", fontsize=10, color=TEAL)
    ax.text(5.48, 1.98, "construction can create concepts absent from traces", fontsize=9.2, color=MUTED)

    ax.text(9.99, 4.30, "formation and acceptance experiences stay separate", fontsize=9.5, color=MUTED)
    chip(ax, 10.02, 3.72, 3.02, "formation error + acceptance error", color=LIGHT, fontsize=9.4)
    chip(ax, 10.02, 3.19, 3.02, "complexity + specificity + abstention", color=LIGHT, fontsize=9.4)
    rounded(ax, 10.02, 2.19, 3.02, 0.73, color="#E8F7F4", edge="#BFE6DE", radius=0.12)
    ax.text(11.53, 2.63, "concept = red AND square", fontsize=10.8, color=NAVY, fontweight="bold", ha="center")
    ax.text(11.53, 2.36, "new object → executable label prediction", fontsize=9.4, color=TEAL, ha="center")
    ax.text(10.02, 1.98, "evaluate on fresh queries after memory formation", fontsize=9.2, color=MUTED)

    arrow(ax, (4.42, 3.48), (5.00, 3.48))
    arrow(ax, (8.96, 3.48), (9.54, 3.48))
    rounded(ax, 0.64, 0.50, 12.73, 0.78, color=NAVY, edge=NAVY, radius=0.13)
    ax.text(
        7.0, 0.94, "Candidate generation ≠ candidate acceptance",
        color="white", fontsize=14, fontweight="bold", ha="center",
    )
    ax.text(
        7.0, 0.68,
        "Candidate coverage is not correctness. Real experiences determine which executable concepts survive.",
        color="#DDE6F4", fontsize=10.3, ha="center",
    )
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
