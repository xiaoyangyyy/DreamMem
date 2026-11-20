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


def architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.62, 10.03, "DreamMem system architecture", fontsize=24, fontweight="bold", color=NAVY)
    ax.text(
        0.64, 9.61,
        "The backbone is frozen during offline memory formation; train, validation, and test episodes remain separated.",
        fontsize=11.5, color=MUTED,
    )

    rounded(ax, 0.55, 6.84, 14.90, 2.38, color="#F8FAFC", edge=BORDER, radius=0.18)
    ax.text(
        0.82, 8.93, "A   MASKED DIFFUSION BACKBONE + REVISABLE TRAJECTORY COLLECTION",
        fontsize=10.5, color=BLUE, fontweight="bold",
    )
    for x, width in ((0.86, 3.64), (5.02, 3.76), (9.31, 5.10)):
        rounded(ax, x, 7.12, width, 1.52, color="white", edge=BORDER, radius=0.12)
    ax.text(1.10, 8.31, "Encoded episode", fontsize=13, color=NAVY, fontweight="bold")
    ax.text(1.10, 7.91, "support object attributes + labels", fontsize=9.8, color=MUTED)
    ax.text(1.10, 7.59, "3 concept slots + 4 query labels  [MASK]", fontsize=9.4, color=MUTED)
    ax.text(1.10, 7.29, "dreammem/dcoder_text.py", fontsize=8.7, color=TEAL)

    ax.text(5.27, 8.31, "Masked diffusion backbone", fontsize=12.2, color=NAVY, fontweight="bold")
    ax.text(5.27, 7.93, "Open-dCoder 0.5B  |  nanoDiff 203M  |  small 6.6M", fontsize=9.0, color=MUTED)
    ax.text(5.27, 7.61, "token embeddings → Transformer blocks → logits", fontsize=9.1, color=MUTED)
    ax.text(5.27, 7.29, "absorbing-mask denoising • BF16", fontsize=8.9, color=TEAL)

    ax.text(9.57, 8.31, "Revisable decoder", fontsize=13, color=NAVY, fontweight="bold")
    ax.text(9.57, 7.93, "16 steps × 6 samples × 3 support prefixes = 288 forwards", fontsize=9.4, color=MUTED)
    ax.text(9.57, 7.61, "confidence commitment + low-confidence remasking", fontsize=9.4, color=MUTED)
    ax.text(9.57, 7.29, "dreammem/trajectory.py", fontsize=8.7, color=TEAL)
    arrow(ax, (4.58, 7.88), (4.94, 7.88))
    arrow(ax, (8.86, 7.88), (9.23, 7.88))

    arrow(ax, (11.86, 7.08), (11.86, 6.67))
    rounded(ax, 0.88, 6.01, 14.22, 0.62, color="#EEF2FF", edge="#CDD8F5", radius=0.11)
    ax.text(
        7.99, 6.32,
        "EpisodeTrajectory   •   concept tokens / canonical codes / confidence   •   query tokens / confidence   •   support stage",
        fontsize=10.2, color=NAVY, ha="center", va="center",
    )

    rounded(ax, 0.70, 3.23, 7.05, 2.37, color="#F8FAFC", edge=BORDER, radius=0.18)
    rounded(ax, 8.25, 3.23, 7.05, 2.37, color="#F8FAFC", edge=BORDER, radius=0.18)
    ax.text(0.98, 5.28, "B1   CANDIDATE CONSTRUCTION", fontsize=10.5, color=TEAL, fontweight="bold")
    ax.text(8.53, 5.28, "B2   EVIDENCE-BASED ACCEPTANCE", fontsize=10.5, color=TEAL, fontweight="bold")
    arrow(ax, (4.18, 5.97), (4.18, 5.65))
    arrow(ax, (11.82, 5.97), (11.82, 5.65))

    for x, text in ((1.00, "Observed +\nrevised states"), (3.23, "Symbolic\nneighborhoods"), (5.46, "Search\npolicies")):
        rounded(ax, x, 4.30, 1.98, 0.72, color="white", edge=BORDER, radius=0.10)
        ax.text(x + 0.99, 4.66, text, fontsize=10.0, color=NAVY, ha="center", va="center", linespacing=1.15)
    ax.text(1.01, 3.94, "add • remove • substitute • union / intersection", fontsize=9.3, color=MUTED)
    ax.text(1.01, 3.59, "closure • random • learned • greedy • beam • A*", fontsize=9.3, color=MUTED)
    ax.text(5.53, 3.36, "dreammem/constructive.py", fontsize=8.5, color=TEAL, ha="right")

    rounded(ax, 8.56, 4.30, 2.86, 0.72, color="white", edge=BORDER, radius=0.10)
    rounded(ax, 11.68, 4.30, 3.30, 0.72, color="white", edge=BORDER, radius=0.10)
    ax.text(9.99, 4.66, "Formation\nexamples", fontsize=10.0, color=NAVY, ha="center", va="center")
    ax.text(13.33, 4.66, "Held-out acceptance\nexamples", fontsize=10.0, color=NAVY, ha="center", va="center")
    ax.text(8.57, 3.94, "formation error + acceptance error", fontsize=9.4, color=MUTED)
    ax.text(8.57, 3.62, "complexity • specificity • generation penalty", fontsize=9.3, color=MUTED)
    ax.text(14.93, 3.36, "truncated posterior + MAP / abstention", fontsize=8.9, color=TEAL, ha="right")
    arrow(ax, (7.82, 4.43), (8.17, 4.43))

    arrow(ax, (11.82, 3.18), (11.82, 2.82))
    rounded(ax, 0.70, 1.29, 14.60, 1.48, color="#F8FAFC", edge=BORDER, radius=0.18)
    ax.text(0.98, 2.47, "C   EXECUTABLE MEMORY + FRESH QUERY EVALUATION", fontsize=10.5, color=BLUE, fontweight="bold")
    outputs = (
        (1.00, 3.85, "MAP concept / posterior / abstention"),
        (5.24, 4.18, "Executable labels on new objects"),
        (9.81, 5.18, "Recall • exact recovery • NLL • Brier • AURC • cost"),
    )
    for x, width, text in outputs:
        rounded(ax, x, 1.55, width, 0.62, color="white", edge=BORDER, radius=0.10)
        ax.text(x + width / 2, 1.86, text, fontsize=9.6, color=NAVY, ha="center", va="center")

    rounded(ax, 0.70, 0.25, 14.60, 0.70, color="#FFF7E8", edge="#F1D49A", radius=0.13)
    ax.text(0.98, 0.72, "TRAIN / VALIDATION ONLY", fontsize=9.6, color="#A86700", fontweight="bold")
    ax.text(
        8.00, 0.43,
        "800-step denoising post-training  •  policy fit on train episodes  •  validation selects memory config + budget  •  frozen backbone at test",
        fontsize=9.3, color="#795A25", ha="center",
    )

    svg_path = output / "system_architecture.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(output / "system_architecture.png", dpi=180, bbox_inches="tight", facecolor="white")
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
    architecture(output)
    results_figure(methods, ablations, output)
    print(f"Wrote README figures to {output}")


if __name__ == "__main__":
    main()
