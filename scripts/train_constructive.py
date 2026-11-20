from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config
from dreammem.constructive import evaluate_search, fit_search_policy
from dreammem.evaluation import summarize_results, tuning_objective
from dreammem.experiment import load_trajectories
from dreammem.task_factory import make_task_generator
from dreammem.tasks import TaskGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--train-trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="24,32,48,64,96")
    parser.add_argument("--seed", type=int, default=1700)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = Path(args.evaluation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    memory = MemoryConfig(
        **json.loads((evaluation / "best_memory_configs.json").read_text(encoding="utf-8"))["evidence_guided"]
    )
    generator = make_task_generator(config)
    training = load_trajectories(args.train_trajectories)
    policy = fit_search_policy(training, generator.library, args.seed)
    with (output / "policy.pkl").open("wb") as handle:
        pickle.dump(policy, handle)
    validation = load_trajectories(evaluation / "trajectories" / "val.pt")
    validation += load_trajectories(evaluation / "trajectories" / "val_compositional.pt")
    budgets = [int(value) for value in args.budgets.split(",")]
    all_results = []
    all_diagnostics = []
    for strategy in ("random_search", "learned_search"):
        for budget in budgets:
            results, diagnostics = evaluate_search(
                validation,
                memory,
                generator.library,
                policy if strategy == "learned_search" else None,
                budget,
                strategy,
                args.seed + budget,
            )
            results["budget"] = budget
            diagnostics["budget"] = budget
            all_results.append(results)
            all_diagnostics.append(diagnostics)
            summary = summarize_results(results)
            print(strategy, budget, tuning_objective(results, strategy), flush=True)
            print(summary[["split", "concept_accuracy", "query_accuracy", "query_nll"]].to_string(index=False), flush=True)
    closure_results, closure_diagnostics = evaluate_search(
        validation,
        memory,
        generator.library,
        None,
        0,
        "closure_search",
        args.seed,
    )
    closure_results["budget"] = 0
    closure_diagnostics["budget"] = 0
    all_results.append(closure_results)
    all_diagnostics.append(closure_diagnostics)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output / "validation_results.csv", index=False)
    pd.concat(all_diagnostics, ignore_index=True).to_csv(output / "validation_diagnostics.csv", index=False)
    rows = []
    for (method, budget), group in combined.groupby(["method", "budget"]):
        rows.append(
            {
                "method": method,
                "budget": budget,
                "objective": tuning_objective(group, method),
                "concept_accuracy": group["concept_accuracy"].mean(),
                "query_accuracy": group["query_accuracy"].mean(),
                "query_nll": group["query_nll"].mean(),
                "compositional_concept_accuracy": group[group["split"] == "val_compositional"]["concept_accuracy"].mean(),
                "compositional_query_accuracy": group[group["split"] == "val_compositional"]["query_accuracy"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "validation_summary.csv", index=False)
    best = {}
    for method in ("random_search", "learned_search"):
        subset = summary[summary["method"] == method]
        row = subset.loc[subset["objective"].idxmax()]
        best[method] = {"budget": int(row["budget"]), "objective": float(row["objective"])}
    (output / "best_search.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_episodes": len(training),
                "feature_count": int(policy.estimator.n_features_in_),
                "branch": policy.branch,
                "pair_pool": policy.pair_pool,
                "best": best,
                "memory": asdict(memory),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
