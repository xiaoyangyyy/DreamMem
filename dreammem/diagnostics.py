from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import MemoryConfig
from .evaluation import decision_metrics
from .memory import MemoryDecision, MemoryFormer, _error
from .semantics import INVALID_CONCEPT
from .tasks import ConceptLibrary, code_to_concept, concept_to_code, valid_concept
from .trajectory import EpisodeTrajectory, StageTrajectory


def state_candidates(trajectory: EpisodeTrajectory) -> set[int]:
    return {
        int(code)
        for stage in trajectory.stages
        for code in np.unique(stage.concept_codes)
        if int(code) != INVALID_CONCEPT
    }


def final_candidates(trajectory: EpisodeTrajectory) -> set[int]:
    return {
        int(stage.concept_codes[sample, -1])
        for stage in trajectory.stages
        for sample in range(stage.concept_codes.shape[0])
        if int(stage.concept_codes[sample, -1]) != INVALID_CONCEPT
    }


def constructive_closure(
    candidates: set[int],
    statistics: dict | None = None,
    library: ConceptLibrary | None = None,
) -> set[int]:
    library = library or ConceptLibrary()
    closure = set(candidates)
    generated = set()
    generated_instances = 0
    iterations = 0
    changed = True
    while changed:
        changed = False
        iterations += 1
        for left, right in combinations(sorted(closure), 2):
            left_concept = set(library.code_to_concept(left))
            right_concept = set(library.code_to_concept(right))
            for concept in (
                tuple(sorted(left_concept | right_concept)),
                tuple(sorted(left_concept & right_concept)),
            ):
                if not concept or not library.valid_concept(concept):
                    continue
                generated_instances += 1
                code = concept_to_code(concept)
                generated.add(code)
                if code not in closure:
                    closure.add(code)
                    changed = True
    if statistics is not None:
        statistics.update(
            generated_unique_candidates=len(generated),
            generated_candidate_instances=generated_instances,
            closure_iterations=iterations,
        )
    return closure


def candidate_diagnostics(
    trajectories: list[EpisodeTrajectory],
    config: MemoryConfig,
) -> pd.DataFrame:
    rows = []
    for trajectory in trajectories:
        former = MemoryFormer(config)
        states = state_candidates(trajectory)
        finals = final_candidates(trajectory)
        generated = former._revision_scores(trajectory)[1]
        candidates = states | generated
        closure = constructive_closure(states)
        decision = former.form(trajectory, "evidence_guided")
        ordered = sorted(decision.posterior, key=decision.posterior.get, reverse=True)
        true_code = concept_to_code(trajectory.episode.concept)
        rank = ordered.index(true_code) + 1 if true_code in ordered else 0
        oracle = MemoryDecision(
            method="oracle_memory",
            concept=trajectory.episode.concept,
            posterior={true_code: 1.0},
            candidate_count=1,
            validation_error=_error(true_code, trajectory.episode, 6),
            novel_candidate=False,
        )
        oracle_metrics = decision_metrics(oracle, trajectory)
        rows.append(
            {
                "split": trajectory.episode.split,
                "episode": trajectory.episode.index,
                "true_code": true_code,
                "final_recall": float(true_code in finals),
                "state_recall": float(true_code in states),
                "revision_recall": float(true_code in candidates),
                "closure_recall": float(true_code in closure),
                "recall_at_1": float(rank == 1),
                "recall_at_5": float(0 < rank <= 5),
                "recall_at_20": float(0 < rank <= 20),
                "recall_at_48": float(rank > 0),
                "true_rank": rank,
                "rank_accuracy_given_present": float(rank == 1) if true_code in candidates else np.nan,
                "true_posterior": decision.posterior.get(true_code, 0.0),
                "state_candidates": len(states),
                "revision_candidates": len(candidates),
                "closure_candidates": len(closure),
                "true_validation_error": _error(true_code, trajectory.episode, 6),
                "predicted_validation_error": decision.validation_error,
                "oracle_query_accuracy": oracle_metrics["query_accuracy"],
                "oracle_query_nll": oracle_metrics["query_nll"],
            }
        )
    return pd.DataFrame(rows)


def shuffled_trajectory(
    trajectory: EpisodeTrajectory,
    rng: np.random.Generator,
) -> EpisodeTrajectory:
    stages = []
    for stage in trajectory.stages:
        permutation = np.asarray(
            [rng.permutation(stage.concept_codes.shape[1]) for _ in range(stage.concept_codes.shape[0])]
        )
        stages.append(
            StageTrajectory(
                support_count=stage.support_count,
                concept_tokens=np.take_along_axis(stage.concept_tokens, permutation[:, :, None], axis=1),
                concept_codes=np.take_along_axis(stage.concept_codes, permutation, axis=1),
                concept_confidence=np.take_along_axis(stage.concept_confidence, permutation[:, :, None], axis=1),
                query_tokens=np.take_along_axis(stage.query_tokens, permutation[:, :, None], axis=1),
                query_confidence=np.take_along_axis(stage.query_confidence, permutation[:, :, None], axis=1),
            )
        )
    return EpisodeTrajectory(trajectory.episode, stages)


def order_ablation(
    trajectories: list[EpisodeTrajectory],
    configs: dict[str, MemoryConfig],
    shuffles: int = 8,
    seed: int = 1900,
) -> pd.DataFrame:
    methods = ("bag_of_states", "revision_dynamics", "evidence_states", "evidence_guided")
    rows = []
    for trajectory in trajectories:
        for shuffle in range(-1, shuffles):
            rng = np.random.default_rng(seed + 1009 * shuffle + trajectory.episode.index)
            current = trajectory if shuffle < 0 else shuffled_trajectory(trajectory, rng)
            for method in methods:
                decision = MemoryFormer(configs[method]).form(current, method)
                metrics = decision_metrics(decision, current)
                rows.append(
                    {
                        "split": trajectory.episode.split,
                        "episode": trajectory.episode.index,
                        "shuffle": shuffle,
                        "method": method,
                        "concept_accuracy": metrics["concept_accuracy"],
                        "query_accuracy": metrics["query_accuracy"],
                        "query_nll": metrics["query_nll"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_candidate_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_recall",
        "state_recall",
        "revision_recall",
        "closure_recall",
        "recall_at_1",
        "recall_at_5",
        "recall_at_20",
        "recall_at_48",
        "rank_accuracy_given_present",
        "true_posterior",
        "state_candidates",
        "revision_candidates",
        "closure_candidates",
        "true_validation_error",
        "predicted_validation_error",
        "oracle_query_accuracy",
        "oracle_query_nll",
    ]
    return results.groupby("split")[metrics].mean().reset_index()


def summarize_order_ablation(results: pd.DataFrame) -> pd.DataFrame:
    original = results[results["shuffle"] == -1].set_index(["split", "episode", "method"])
    rows = []
    for (split, method), group in results[results["shuffle"] >= 0].groupby(["split", "method"]):
        shuffled = group.groupby("episode")[["concept_accuracy", "query_accuracy", "query_nll"]].mean()
        baseline = original.loc[(split, slice(None), method), ["concept_accuracy", "query_accuracy", "query_nll"]]
        baseline.index = baseline.index.droplevel([0, 2])
        difference = shuffled - baseline
        rows.append(
            {
                "split": split,
                "method": method,
                "original_concept_accuracy": baseline["concept_accuracy"].mean(),
                "shuffled_concept_accuracy": shuffled["concept_accuracy"].mean(),
                "concept_accuracy_change": difference["concept_accuracy"].mean(),
                "original_query_accuracy": baseline["query_accuracy"].mean(),
                "shuffled_query_accuracy": shuffled["query_accuracy"].mean(),
                "query_accuracy_change": difference["query_accuracy"].mean(),
                "original_query_nll": baseline["query_nll"].mean(),
                "shuffled_query_nll": shuffled["query_nll"].mean(),
                "query_nll_change": difference["query_nll"].mean(),
                "shuffles": int(group["shuffle"].nunique()),
            }
        )
    return pd.DataFrame(rows)
