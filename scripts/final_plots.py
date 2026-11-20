from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")


METHODS = (
    "final",
    "evidence_guided",
    "closure_search",
    "random_search_matched",
    "random_search",
    "learned_search",
    "symbolic",
)
LABELS = {
    "final": "Final",
    "evidence_guided": "Evidence",
    "closure_search": "Closure",
    "random_search_matched": "Random matched",
    "random_search": "Random 96",
    "learned_search": "Learned search",
    "symbolic": "Symbolic",
}


def facet_grid(count: int, width: float = 5.0):
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, 4 * rows), squeeze=False)
    return figure, axes.ravel(), rows, columns


def concept_accuracy_figures(metrics: pd.DataFrame, output: Path) -> None:
    for task in metrics.loc[metrics["method"] == "learned_search", "task"].unique():
        subset = metrics[(metrics["task"] == task) & metrics["method"].isin(METHODS)]
        splits = list(dict.fromkeys(subset["split"]))
        figure, axes, rows, columns = facet_grid(len(splits))
        for axis, split in zip(axes, splits):
            current = subset[subset["split"] == split].set_index("method").reindex(METHODS)
            x = np.arange(len(METHODS))
            errors = current["concept_accuracy_std"].fillna(0).to_numpy()
            axis.bar(x, current["concept_accuracy_mean"], yerr=errors, capsize=3)
            axis.set_xticks(x, [LABELS[method] for method in METHODS], rotation=35, ha="right")
            axis.set_ylim(0, 1.05)
            axis.set_title(split)
            axis.set_ylabel("Exact concept recovery")
            axis.grid(axis="y", alpha=0.2)
        for axis in axes[len(splits):]:
            axis.set_visible(False)
        figure.suptitle(f"{task}: candidate construction and acceptance")
        figure.tight_layout(rect=(0, 0, 1, 0.98 if rows > 1 else 0.95))
        figure.savefig(output / f"{task}_concept_accuracy.png", dpi=180)
        plt.close(figure)


def difficulty_figure(difficulty: pd.DataFrame, output: Path) -> None:
    for task in difficulty["task"].unique():
        task_data = difficulty[(difficulty["task"] == task) & difficulty["method"].isin(METHODS)]
        splits = list(dict.fromkeys(task_data["split"]))
        figure, axes, rows, _ = facet_grid(len(splits))
        for axis, split in zip(axes, splits):
            subset = task_data[task_data["split"] == split]
            for method in METHODS[1:]:
                current = subset[subset["method"] == method].sort_values("true_size")
                if current.empty:
                    continue
                axis.plot(current["true_size"], current["concept_accuracy_mean"], marker="o", label=LABELS[method])
            axis.set_xticks(sorted(subset["true_size"].unique()))
            axis.set_ylim(0, 1.05)
            axis.set_xlabel("True concept size")
            axis.set_ylabel("Exact concept recovery")
            axis.set_title(split)
            axis.grid(alpha=0.2)
        for axis in axes[len(splits):]:
            axis.set_visible(False)
        axes[0].legend(fontsize=8)
        figure.suptitle(task)
        figure.tight_layout(rect=(0, 0, 1, 0.98 if rows > 1 else 0.95))
        figure.savefig(output / f"{task}_concept_difficulty.png", dpi=180)
        plt.close(figure)


def cost_figure(metrics: pd.DataFrame, costs: pd.DataFrame, output: Path) -> None:
    tasks = list(metrics.loc[(metrics["method"] == "learned_search") & metrics["task"].isin(costs["task"].unique()), "task"].unique())
    figure, axes, _, _ = facet_grid(len(tasks), width=6.0)
    methods = ("closure_search", "random_search_matched", "random_search", "learned_search", "greedy_search", "beam_search", "symbolic")
    for axis, task in zip(axes, tasks):
        current_metrics = metrics[(metrics["task"] == task) & (metrics["split"] == "compositional")]
        current_costs = costs[(costs["task"] == task) & (costs["split"] == "compositional")]
        merged = current_metrics.merge(current_costs, on=["task", "split", "method"])
        merged = merged[merged["method"].isin(methods)]
        offsets = {"greedy_search": (5, 6), "beam_search": (5, -12), "learned_search": (5, -12)}
        for _, row in merged.iterrows():
            label = LABELS.get(row["method"], row["method"].replace("_search", ""))
            axis.scatter(row["evidence_scored_candidates_mean"], row["concept_accuracy_mean"])
            axis.annotate(
                label,
                (row["evidence_scored_candidates_mean"], row["concept_accuracy_mean"]),
                xytext=offsets.get(row["method"], (5, 5)),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Evidence-scored candidates per episode")
        axis.set_ylabel("Compositional concept recovery")
        axis.set_title(task)
        axis.grid(alpha=0.2)
    for axis in axes[len(tasks):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output / "search_cost_tradeoff.png", dpi=180)
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs" / "final_statistics"
    metrics = pd.read_csv(source / "method_metrics.csv")
    difficulty = pd.read_csv(source / "difficulty_metrics.csv")
    costs = pd.read_csv(source / "compute_costs.csv")
    ablations = pd.read_csv(source / "ablation_metrics.csv")
    cost_metrics = pd.concat(
        [ablations, metrics[metrics["method"] == "random_search"]], ignore_index=True
    )
    concept_accuracy_figures(metrics, source)
    difficulty_figure(difficulty, source)
    cost_figure(cost_metrics, costs, source)
    print(f"Wrote final figures to {source}")


if __name__ == "__main__":
    main()
