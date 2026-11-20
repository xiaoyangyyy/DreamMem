from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_project import TASKS


METRICS = ("concept_accuracy", "query_accuracy", "query_nll", "brier", "aurc")


def load_results(root: Path) -> pd.DataFrame:
    frames = []
    for task, seeds in TASKS.items():
        for seed in seeds:
            path = root / "outputs" / "constructive_evaluation" / task / f"seed_{seed}" / "evaluation" / "results_all.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["task"] = task
            frame["run_seed"] = seed
            frames.append(frame)
    for mode in ("ar", "mlm"):
        path = root / "outputs" / "decoder_evaluation" / "dcoder" / "seed_17" / mode / "results_all.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["task"] = f"dcoder_{mode}"
            frame["run_seed"] = 17
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No completed evaluation results")
    return pd.concat(frames, ignore_index=True)


def load_ablation_results(root: Path) -> pd.DataFrame:
    frames = []
    for task, seeds in TASKS.items():
        for seed in seeds:
            path = root / "outputs" / "constructive_evaluation" / task / f"seed_{seed}" / "evaluation" / "ablations" / "ablation_results.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame["task"] = task
            frame["run_seed"] = seed
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def metric_summary(results: pd.DataFrame) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in results]
    per_seed = results.groupby(["task", "split", "method", "run_seed"])[available].mean().reset_index()
    aggregate = per_seed.groupby(["task", "split", "method"])[available].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    aggregate = aggregate.reset_index()
    counts = per_seed.groupby(["task", "split", "method"]).size().rename("seeds").reset_index()
    return aggregate.merge(counts, on=["task", "split", "method"])


def concept_summaries(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["concept_accuracy", "query_accuracy", "query_nll"]
    concept = (
        results.groupby(["task", "split", "method", "true_code", "true_size"])[metrics]
        .agg(["mean", "std", "count"])
    )
    concept.columns = [f"{metric}_{statistic}" for metric, statistic in concept.columns]
    concept = concept.reset_index()
    per_seed = (
        results.groupby(["task", "split", "method", "true_size", "run_seed"])[metrics]
        .mean()
        .reset_index()
    )
    difficulty = per_seed.groupby(["task", "split", "method", "true_size"])[metrics].agg(["mean", "std"])
    difficulty.columns = [f"{metric}_{statistic}" for metric, statistic in difficulty.columns]
    return concept, difficulty.reset_index()


def paired_effects(results: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    neural = results[results["method"] == "learned_search"]
    for (task, split), reference in neural.groupby(["task", "split"]):
        subset = results[(results["task"] == task) & (results["split"] == split)]
        for method in sorted(subset["method"].unique()):
            if method == "learned_search":
                continue
            other = subset[subset["method"] == method]
            merged = reference.merge(
                other,
                on=["run_seed", "episode"],
                suffixes=("_reference", "_other"),
            )
            for metric in ("concept_accuracy", "query_accuracy", "query_nll"):
                per_seed = []
                for _, group in merged.groupby("run_seed"):
                    difference = (
                        group[f"{metric}_reference"].to_numpy(dtype=float)
                        - group[f"{metric}_other"].to_numpy(dtype=float)
                    )
                    per_seed.append(difference)
                estimate = float(np.mean([values.mean() for values in per_seed]))
                estimates = []
                for _ in range(samples):
                    estimates.append(
                        float(
                            np.mean(
                                [
                                    values[rng.integers(0, len(values), len(values))].mean()
                                    for values in per_seed
                                ]
                            )
                        )
                    )
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "reference": "learned_search",
                        "baseline": method,
                        "metric": metric,
                        "difference": estimate,
                        "low": float(np.quantile(estimates, 0.025)),
                        "high": float(np.quantile(estimates, 0.975)),
                        "seeds": len(per_seed),
                        "episodes_per_seed": int(merged.groupby("run_seed").size().min()),
                    }
                )
    return pd.DataFrame(rows)


def compute_summary(root: Path) -> pd.DataFrame:
    frames = []
    for task, seeds in TASKS.items():
        for seed in seeds:
            base = root / "outputs" / "constructive_evaluation" / task / f"seed_{seed}" / "evaluation"
            path = base / "ablations" / "ablation_costs.csv"
            runtime_path = base / "runtime.json"
            if not path.exists() or not runtime_path.exists():
                continue
            frame = pd.read_csv(path)
            frame["task"] = task
            frame["run_seed"] = seed
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            frame["dllm_forward_passes"] = runtime["forward_passes_per_episode"]
            frames.append(frame)
            audit_path = base / "audit" / "audit_costs.csv"
            if audit_path.exists():
                audit = pd.read_csv(audit_path)
                audit = audit[audit["method"] == "random_search"].copy()
                audit["task"] = task
                audit["run_seed"] = seed
                audit["dllm_forward_passes"] = runtime["forward_passes_per_episode"]
                frames.append(audit)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    excluded = {"episode", "seed", "run_seed"}
    numeric = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in excluded
    ]
    per_seed = combined.groupby(["task", "split", "method", "run_seed"])[numeric].mean().reset_index()
    aggregate = per_seed.groupby(["task", "split", "method"])[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{statistic}" for column, statistic in aggregate.columns]
    return aggregate.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = load_results(root)
    if args.final:
        expected = sum(len(seeds) for seeds in TASKS.values()) + 2
        assert results.groupby(["task", "run_seed"]).ngroups == expected
    output = root / "outputs" / "final_statistics"
    output.mkdir(parents=True, exist_ok=True)
    metric_summary(results).to_csv(output / "method_metrics.csv", index=False)
    ablations = load_ablation_results(root)
    if not ablations.empty:
        metric_summary(ablations).to_csv(output / "ablation_metrics.csv", index=False)
        ablations.to_csv(output / "all_ablation_episode_results.csv", index=False)
    if args.final:
        assert ablations.groupby(["task", "run_seed"]).ngroups == expected - 2
    concepts, difficulty = concept_summaries(results)
    concepts.to_csv(output / "concept_metrics.csv", index=False)
    difficulty.to_csv(output / "difficulty_metrics.csv", index=False)
    effects = paired_effects(results, args.samples, args.seed)
    effects.to_csv(output / "paired_effects.csv", index=False)
    compute_summary(root).to_csv(output / "compute_costs.csv", index=False)
    results.to_csv(output / "all_episode_results.csv", index=False)
    metadata = {
        "runs": int(results.groupby(["task", "run_seed"]).ngroups),
        "rows": len(results),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "paired_interval_note": "Episodes are resampled within each seed; seed-level estimates are equally weighted.",
        "final": args.final,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
