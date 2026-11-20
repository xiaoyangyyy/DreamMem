from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_summary(summary: pd.DataFrame, destination: Path) -> None:
    splits = list(dict.fromkeys(summary["split"]))
    methods = list(dict.fromkeys(summary["method"]))
    figure, axes = plt.subplots(2, len(splits), figsize=(6 * len(splits), 8), squeeze=False)
    x = np.arange(len(methods))
    for column, split in enumerate(splits):
        subset = summary[summary["split"] == split].set_index("method")
        for row, metric in enumerate(("concept_accuracy", "query_accuracy")):
            axis = axes[row, column]
            values = [float(subset.loc[method, metric]) for method in methods]
            axis.bar(x, values, color="steelblue" if row == 0 else "darkorange")
            axis.set_title(f"{split}: {metric}")
            axis.set_ylim(0, 1)
            axis.set_xticks(x, methods, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_risk_coverage(results: pd.DataFrame, destination: Path) -> None:
    splits = list(dict.fromkeys(results["split"]))
    figure, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for axis, split in zip(axes[0], splits):
        subset = results[results["split"] == split]
        for method, group in subset.groupby("method", sort=False):
            ordered = group.sort_values("confidence", ascending=False)
            coverages = np.linspace(0.1, 1.0, 10)
            risks = []
            for coverage in coverages:
                count = max(1, int(np.ceil(len(ordered) * coverage)))
                risks.append(1.0 - float(ordered.iloc[:count]["query_accuracy"].mean()))
            axis.plot(coverages, risks, marker="o", label=method)
        axis.set_title(split)
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Query error")
        axis.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_training(history_path: Path, destination: Path) -> None:
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["train_loss"], label="train")
    axes[0].plot(history["step"], history["val_loss"], label="validation")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Masked cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(history["step"], history["val_query_accuracy"], label="query")
    axes[1].plot(history["step"], history["val_concept_accuracy"], label="concept")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
