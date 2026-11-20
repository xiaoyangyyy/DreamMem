from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def value(row, metric):
    mean = row[f"{metric}_mean"]
    deviation = row.get(f"{metric}_std", float("nan"))
    if pd.isna(deviation):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{deviation:.4f}"


def table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main_table(metrics):
    methods = ["evidence_guided", "closure_search", "random_search_matched", "random_search", "learned_search", "symbolic"]
    rows = []
    for task in ("dcoder", "relational"):
        for split in ("seen", "unseen", "compositional"):
            for method in methods:
                selected = metrics[(metrics.task == task) & (metrics.split == split) & (metrics.method == method)]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                rows.append([task, split, method, value(row, "concept_accuracy"), value(row, "query_accuracy"), value(row, "query_nll")])
    return table(rows, ["Task", "Split", "Method", "Concept", "Query", "NLL"])


def ablation_table(ablations):
    rows = []
    for task in ablations.task.unique():
        for split in ablations[ablations.task == task].split.unique():
            for method in ablations[(ablations.task == task) & (ablations.split == split)].method.unique():
                selected = ablations[(ablations.task == task) & (ablations.split == split) & (ablations.method == method)]
                row = selected.iloc[0]
                rows.append([task, split, method, value(row, "concept_accuracy"), value(row, "query_accuracy"), value(row, "query_nll")])
    return table(rows, ["Task", "Split", "Method", "Concept", "Query", "NLL"])


def decoder_table(root, metrics):
    rows = []
    for task, mode, passes in (("dcoder", "dLLM", 288), ("dcoder_ar", "AR", 294), ("dcoder_mlm", "MLM", 18)):
        for split in ("seen", "unseen", "compositional"):
            for method in ("evidence_guided", "learned_search"):
                selected = metrics[(metrics.task == task) & (metrics.split == split) & (metrics.method == method)]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                seconds = ""
                if mode != "dLLM":
                    runtime_path = root / "outputs" / "decoder_evaluation" / "dcoder" / "seed_17" / mode.lower() / "runtime.json"
                    if runtime_path.exists():
                        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                        seconds = f"{runtime['trajectory_seconds'].get(split, float('nan')):.2f}"
                rows.append([mode, split, method, value(row, "concept_accuracy"), value(row, "query_accuracy"), value(row, "query_nll"), passes, seconds])
    return table(rows, ["Decoder", "Split", "Method", "Concept", "Query", "NLL", "Forward passes", "Trajectory seconds"])


def extended_table(metrics):
    methods = ["evidence_guided", "closure_search", "random_search_matched", "random_search", "learned_search", "symbolic"]
    rows = []
    excluded = {"dcoder", "relational", "dcoder_ar", "dcoder_mlm"}
    for task in metrics.loc[~metrics.task.isin(excluded), "task"].unique():
        for split in metrics[metrics.task == task].split.unique():
            for method in methods:
                selected = metrics[(metrics.task == task) & (metrics.split == split) & (metrics.method == method)]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                rows.append([task, split, method, value(row, "concept_accuracy"), value(row, "query_accuracy"), value(row, "query_nll")])
    return table(rows, ["Task", "Split", "Method", "Concept", "Query", "NLL"])


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs" / "final_statistics"
    metrics = pd.read_csv(source / "method_metrics.csv")
    ablations = pd.read_csv(source / "ablation_metrics.csv")
    sections = ["# Automated paper tables", "## Main results", main_table(metrics), "## Extended tasks", extended_table(metrics), "## Full ablations", ablation_table(ablations)]
    decoders = decoder_table(root, metrics)
    if decoders.splitlines()[2:]:
        sections.extend(["## Decoder controls", decoders])
    destination = source / "paper_tables.md"
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
