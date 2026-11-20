from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config
from dreammem.constructive import evaluate_search, fit_search_policy
from dreammem.experiment import load_trajectories
from dreammem.task_factory import make_task_generator
from dreammem.search_baselines import evaluate_classical_search
from dreammem.tasks import TaskGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--train-trajectories", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = Path(args.evaluation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "policies").mkdir(exist_ok=True)
    with (Path(args.policy) / "policy.pkl").open("rb") as handle:
        full_policy = pickle.load(handle)
    search = json.loads((Path(args.policy) / "best_search.json").read_text(encoding="utf-8"))
    raw = json.loads((Path(args.baseline_evaluation) / "best_memory_configs.json").read_text(encoding="utf-8"))
    memory = MemoryConfig(**raw["evidence_guided"])
    generator = make_task_generator(config)
    training = load_trajectories(args.train_trajectories)
    variants = {
        "no_evidence": ("no_evidence", "hgb"),
        "no_trajectory": ("no_trajectory", "hgb"),
        "logistic": ("full", "logistic"),
        "mlp": ("full", "mlp"),
    }
    policies = {}
    metadata = {}
    for name, (feature_mode, estimator_kind) in variants.items():
        print(f"Fitting {name}", flush=True)
        policies[name] = fit_search_policy(
            training,
            generator.library,
            1700,
            feature_mode=feature_mode,
            estimator_kind=estimator_kind,
        )
        with (output / "policies" / f"{name}.pkl").open("wb") as handle:
            pickle.dump(policies[name], handle)
        metadata[name] = {"feature_mode": feature_mode, "estimator_kind": estimator_kind}
    (output / "variants.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    budget = search["best"]["learned_search"]["budget"]
    methods = [
        ("learned_search", full_policy),
        ("learned_search_atoms", full_policy),
        ("learned_search_no_merge", full_policy),
        ("learned_search_no_evidence_features", policies["no_evidence"]),
        ("learned_search_no_trajectory_features", policies["no_trajectory"]),
        ("learned_search_logistic", policies["logistic"]),
        ("learned_search_mlp", policies["mlp"]),
        ("random_search_matched", None),
    ]
    all_results = []
    all_costs = []
    for split in config.eval.splits:
        trajectories = load_trajectories(evaluation / "trajectories" / f"{split}.pt")
        for method, policy in methods:
            results, costs = evaluate_search(
                trajectories,
                memory,
                generator.library,
                policy,
                budget,
                method,
                18000 + budget,
            )
            all_results.append(results)
            all_costs.append(costs)
        results, costs = evaluate_search(
            trajectories, memory, generator.library, None, 0, "closure_search", 18000
        )
        all_results.append(results)
        all_costs.append(costs)
        for method in ("greedy_search", "beam_search", "astar_search"):
            results, costs = evaluate_classical_search(
                trajectories, memory, generator.library, budget, method
            )
            all_results.append(results)
            all_costs.append(costs)
    results = pd.concat(all_results, ignore_index=True)
    costs = pd.concat(all_costs, ignore_index=True)
    audit_results = pd.read_csv(evaluation / "audit" / "audit_results.csv")
    audit_costs = pd.read_csv(evaluation / "audit" / "audit_costs.csv")
    direct_methods = (
        "learned_ranker_full",
        "learned_ranker_initial",
        "learned_search_policy_rank",
        "evidence_guided",
        "symbolic",
    )
    results = pd.concat(
        [results, audit_results[audit_results["method"].isin(direct_methods)]],
        ignore_index=True,
    )
    costs = pd.concat(
        [costs, audit_costs[audit_costs["method"].isin(direct_methods)]],
        ignore_index=True,
    )
    results.to_csv(output / "ablation_results.csv", index=False)
    costs.to_csv(output / "ablation_costs.csv", index=False)
    metrics = ["concept_accuracy", "query_accuracy", "query_nll"]
    summary = results.groupby(["split", "method"])[metrics].mean().reset_index()
    cost_columns = [
        "candidate_recall",
        "candidate_count",
        "generated_unique_candidates",
        "generated_candidate_instances",
        "policy_scored_instances",
        "evidence_scored_candidates",
        "expansions",
        "search_seconds",
        "evidence_seconds",
    ]
    summary = summary.merge(costs.groupby(["split", "method"])[cost_columns].mean().reset_index(), on=["split", "method"])
    summary.to_csv(output / "ablation_summary.csv", index=False)
    original = pd.read_csv(evaluation / "results_all.csv")
    for metric in metrics:
        expected = original[original["method"] == "learned_search"].sort_values(["split", "episode"])[metric]
        actual = results[results["method"] == "learned_search"].sort_values(["split", "episode"])[metric]
        assert np.allclose(actual.to_numpy(), expected.to_numpy()), metric
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
