from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config
from dreammem.constructive import (
    evaluate_policy_ranker,
    evaluate_search,
    evidence_decision,
    initial_candidates,
)
from dreammem.evaluation import decision_metrics
from dreammem.experiment import load_trajectories
from dreammem.memory import symbolic_memory
from dreammem.task_factory import make_task_generator
from dreammem.tasks import TaskGenerator, concept_to_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = Path(args.evaluation)
    config.eval.seed = json.loads((evaluation / "runtime.json").read_text(encoding="utf-8"))["episode_seed"]
    output = Path(args.output) if args.output else evaluation / "audit"
    output.mkdir(parents=True, exist_ok=True)
    with (Path(args.policy) / "policy.pkl").open("rb") as handle:
        policy = pickle.load(handle)
    search = json.loads((Path(args.policy) / "best_search.json").read_text(encoding="utf-8"))
    raw = json.loads((Path(args.baseline_evaluation) / "best_memory_configs.json").read_text(encoding="utf-8"))
    memory = MemoryConfig(**raw["evidence_guided"])
    symbolic_config = MemoryConfig(**raw.get("symbolic", raw["evidence_guided"]))
    generator = make_task_generator(config)
    original = pd.read_csv(evaluation / "results_all.csv", low_memory=False)
    all_results = []
    all_costs = []
    for split in config.eval.splits:
        trajectories = load_trajectories(evaluation / "trajectories" / f"{split}.pt")
        for method, budget in (
            ("closure_search", 0),
            ("random_search", search["best"]["random_search"]["budget"]),
            ("random_search_matched", search["best"]["learned_search"]["budget"]),
            ("learned_search", search["best"]["learned_search"]["budget"]),
        ):
            results, costs = evaluate_search(
                trajectories,
                memory,
                generator.library,
                policy if method == "learned_search" else None,
                budget,
                method,
                config.eval.seed + 900 + budget,
            )
            all_results.append(results)
            all_costs.append(costs)
        learned_budget = search["best"]["learned_search"]["budget"]
        for method in ("learned_ranker_full", "learned_ranker_initial", "learned_search_policy_rank"):
            results, costs = evaluate_policy_ranker(
                trajectories,
                memory,
                generator.library,
                policy,
                learned_budget,
                method,
                config.eval.seed + 900 + learned_budget,
            )
            all_results.append(results)
            all_costs.append(costs)
        for trajectory in trajectories:
            started = time.perf_counter()
            candidates = initial_candidates(trajectory, generator.library)
            decision = evidence_decision(trajectory, candidates, memory, "evidence_guided", generator.library)
            elapsed = time.perf_counter() - started
            all_results.append(pd.DataFrame([decision_metrics(decision, trajectory)]))
            all_costs.append(
                {
                    "split": split,
                    "episode": trajectory.episode.index,
                    "method": "evidence_guided",
                    "candidate_recall": float(concept_to_code(trajectory.episode.concept) in candidates),
                    "candidate_count": len(candidates),
                    "initial_candidate_count": len(candidates),
                    "generated_unique_candidates": 0,
                    "generated_candidate_instances": 0,
                    "controller_score_calls": 0,
                    "controller_scored_instances": 0,
                    "controller_scored_unique": 0,
                    "policy_scored_instances": 0,
                    "ranker_scored_candidates": 0,
                    "evidence_scored_candidates": len(candidates),
                    "expansions": 0,
                    "search_budget": len(candidates),
                    "search_seconds": elapsed,
                    "evidence_seconds": elapsed,
                }
            )
            started = time.perf_counter()
            concepts = generator.library.concepts_for_split(split)
            decision = symbolic_memory(trajectory.episode, concepts, symbolic_config, generator.library)
            elapsed = time.perf_counter() - started
            all_results.append(pd.DataFrame([decision_metrics(decision, trajectory)]))
            all_costs.append(
                {
                    "split": split,
                    "episode": trajectory.episode.index,
                    "method": "symbolic",
                    "candidate_recall": 1.0,
                    "candidate_count": len(concepts),
                    "initial_candidate_count": len(concepts),
                    "generated_unique_candidates": 0,
                    "generated_candidate_instances": 0,
                    "controller_score_calls": 0,
                    "controller_scored_instances": 0,
                    "controller_scored_unique": 0,
                    "policy_scored_instances": 0,
                    "ranker_scored_candidates": 0,
                    "evidence_scored_candidates": len(concepts),
                    "expansions": 0,
                    "search_budget": len(concepts),
                    "search_seconds": elapsed,
                    "evidence_seconds": elapsed,
                }
            )
    results = pd.concat(all_results, ignore_index=True)
    costs = pd.concat(
        [frame if isinstance(frame, pd.DataFrame) else pd.DataFrame([frame]) for frame in all_costs],
        ignore_index=True,
    )
    results["seed"] = int(original["seed"].iloc[0])
    costs["seed"] = int(original["seed"].iloc[0])
    costs["dllm_forward_passes"] = (
        config.trajectory.samples
        * config.trajectory.steps
        * len(config.trajectory.support_stages)
    )
    costs["ranker_scored_candidates"] = costs["ranker_scored_candidates"].fillna(0)
    results.to_csv(output / "audit_results.csv", index=False)
    costs.to_csv(output / "audit_costs.csv", index=False)
    metrics = ["concept_accuracy", "query_accuracy", "query_nll"]
    summary = results.groupby(["split", "method"])[metrics].mean().reset_index()
    cost_columns = [
        "candidate_recall",
        "candidate_count",
        "generated_unique_candidates",
        "generated_candidate_instances",
        "controller_score_calls",
        "controller_scored_instances",
        "controller_scored_unique",
        "policy_scored_instances",
        "ranker_scored_candidates",
        "evidence_scored_candidates",
        "expansions",
        "search_seconds",
        "evidence_seconds",
        "dllm_forward_passes",
    ]
    summary = summary.merge(costs.groupby(["split", "method"])[cost_columns].mean().reset_index(), on=["split", "method"])
    summary.to_csv(output / "audit_summary.csv", index=False)
    for method in (
        "learned_search", "random_search", "random_search_matched", "closure_search",
        "evidence_guided", "symbolic",
    ):
        expected = original[original["method"] == method].sort_values(["split", "episode"])
        actual = results[results["method"] == method].sort_values(["split", "episode"])
        for metric in metrics:
            assert np.allclose(actual[metric].to_numpy(), expected[metric].to_numpy()), (method, metric)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
