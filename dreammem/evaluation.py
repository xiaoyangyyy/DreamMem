from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import MemoryConfig
from .memory import MEMORY_METHODS, MemoryDecision, MemoryFormer, symbolic_memory
from .tasks import ConceptLibrary, concept_label
from .trajectory import EpisodeTrajectory


def decision_metrics(
    decision: MemoryDecision,
    trajectory: EpisodeTrajectory,
) -> dict[str, float | int | str]:
    episode = trajectory.episode
    predicted = decision.concept
    true_code = 0
    for attribute in episode.concept:
        true_code |= 1 << attribute
    probabilities = []
    labels = []
    for query in episode.queries:
        positive = float(np.clip(decision.query_probability(query.attributes), 1e-6, 1 - 1e-6))
        probabilities.append(positive)
        labels.append(float(query.label))
    probability = np.asarray(probabilities)
    label = np.asarray(labels)
    query_accuracy = float(np.mean((probability >= 0.5) == label))
    nll = float(-np.mean(label * np.log(probability) + (1 - label) * np.log(1 - probability)))
    brier = float(np.mean((probability - label) ** 2))
    confidence = max(decision.posterior.values(), default=0.0)
    return {
        "split": episode.split,
        "episode": episode.index,
        "method": decision.method,
        "concept_accuracy": float(predicted == episode.concept),
        "query_accuracy": query_accuracy,
        "query_nll": nll,
        "brier": brier,
        "confidence": confidence,
        "validation_error": decision.validation_error,
        "unsupported_memory": float(decision.validation_error > 0),
        "candidate_count": decision.candidate_count,
        "novel_candidate": float(decision.novel_candidate),
        "predicted_size": 0 if predicted is None else len(predicted),
        "true_size": len(episode.concept),
        "true_code": true_code,
        "predicted_code": -1 if predicted is None else sum(1 << value for value in predicted),
    }


def evaluate_trajectories(
    trajectories: list[EpisodeTrajectory],
    configs: dict[str, MemoryConfig],
    library: ConceptLibrary,
    include_symbolic: bool = True,
) -> pd.DataFrame:
    rows = []
    for trajectory in trajectories:
        for method in MEMORY_METHODS:
            decision = MemoryFormer(configs[method], library).form(trajectory, method)
            rows.append(decision_metrics(decision, trajectory))
        if include_symbolic:
            config = configs.get("symbolic", configs[MEMORY_METHODS[0]])
            decision = symbolic_memory(
                trajectory.episode,
                library.concepts_for_split(trajectory.episode.split),
                config,
                library,
            )
            rows.append(decision_metrics(decision, trajectory))
    return pd.DataFrame(rows)


def selective_metrics(group: pd.DataFrame) -> dict[str, float]:
    ordered = group.sort_values("confidence", ascending=False)
    result = {}
    coverages = np.linspace(0.1, 1.0, 10)
    risks = []
    for coverage in coverages:
        count = max(1, int(math.ceil(len(ordered) * coverage)))
        risk = 1.0 - float(ordered.iloc[:count]["query_accuracy"].mean())
        risks.append(risk)
        result[f"risk_at_{coverage:.1f}"] = risk
    result["aurc"] = float(np.trapezoid(risks, coverages))
    return result


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "concept_accuracy",
        "query_accuracy",
        "query_nll",
        "brier",
        "confidence",
        "validation_error",
        "unsupported_memory",
        "candidate_count",
        "novel_candidate",
    ]
    for (split, method), group in results.groupby(["split", "method"], sort=False):
        row: dict[str, float | int | str] = {
            "split": split,
            "method": method,
            "episodes": len(group),
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        row.update(selective_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_intervals(
    results: pd.DataFrame,
    samples: int = 1000,
    seed: int = 44,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (split, method), group in results.groupby(["split", "method"], sort=False):
        n = len(group)
        indices = rng.integers(0, n, size=(samples, n))
        row: dict[str, float | int | str] = {"split": split, "method": method}
        for metric in ("concept_accuracy", "query_accuracy", "query_nll"):
            values = group[metric].to_numpy(dtype=np.float64)
            estimates = values[indices].mean(axis=1)
            row[f"{metric}_low"] = float(np.quantile(estimates, 0.025))
            row[f"{metric}_high"] = float(np.quantile(estimates, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def concept_cluster_intervals(
    results: pd.DataFrame,
    samples: int = 1000,
    seed: int = 46,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (split, method), group in results.groupby(["split", "method"], sort=False):
        clusters = dict(group.groupby("true_code").indices)
        codes = np.asarray(list(clusters))
        estimates = {
            metric: []
            for metric in ("concept_accuracy", "query_accuracy", "query_nll")
        }
        for _ in range(samples):
            selected_codes = rng.choice(codes, size=len(codes), replace=True)
            indices = np.concatenate([clusters[code] for code in selected_codes])
            for metric in estimates:
                estimates[metric].append(float(group[metric].to_numpy()[indices].mean()))
        row: dict[str, float | int | str] = {
            "split": split,
            "method": method,
            "concept_clusters": len(codes),
        }
        for metric, values in estimates.items():
            row[f"{metric}_low"] = float(np.quantile(values, 0.025))
            row[f"{metric}_high"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(
    results: pd.DataFrame,
    reference: str = "revision_dynamics",
    samples: int = 1000,
    seed: int = 45,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split in results["split"].unique():
        subset = results[results["split"] == split]
        reference_rows = subset[subset["method"] == reference].sort_values("episode")
        for method in subset["method"].unique():
            if method == reference:
                continue
            other = subset[subset["method"] == method].sort_values("episode")
            merged = reference_rows.merge(other, on="episode", suffixes=("_reference", "_other"))
            for metric in ("concept_accuracy", "query_accuracy"):
                difference = (
                    merged[f"{metric}_reference"].to_numpy()
                    - merged[f"{metric}_other"].to_numpy()
                )
                n = len(difference)
                indices = rng.integers(0, n, size=(samples, n))
                estimates = difference[indices].mean(axis=1)
                rows.append(
                    {
                        "split": split,
                        "reference": reference,
                        "baseline": method,
                        "metric": metric,
                        "difference": float(difference.mean()),
                        "low": float(np.quantile(estimates, 0.025)),
                        "high": float(np.quantile(estimates, 0.975)),
                        "win_rate": float((difference > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def tuning_objective(results: pd.DataFrame, method: str) -> float:
    group = results[results["method"] == method]

    def score(frame: pd.DataFrame) -> float:
        return float(
            frame["concept_accuracy"].mean()
            + frame["query_accuracy"].mean()
            - 0.2 * frame["query_nll"].mean()
            - 0.05 * frame["unsupported_memory"].mean()
        )

    split_scores = [score(subset) for _, subset in group.groupby("split")]
    return score(group) + 0.25 * min(split_scores)


def tune_memory_configs(
    trajectories: list[EpisodeTrajectory],
    base: MemoryConfig,
    library: ConceptLibrary,
    trials: int,
    seed: int,
    compositional_trajectories: list[EpisodeTrajectory] | None = None,
) -> tuple[dict[str, MemoryConfig], pd.DataFrame]:
    methods = MEMORY_METHODS + ("symbolic",)
    best_configs = {method: base for method in methods}
    best_scores = {method: -float("inf") for method in methods}
    rng = np.random.default_rng(seed)
    history = []
    anchors = [base]
    anchors.extend(
        MemoryConfig(
            validation_weight=4.0,
            complexity_penalty=0.0,
            confidence_weight=0.0,
            frequency_weight=0.0,
            revision_bonus=0.0,
            generalization_bonus=0.0,
            formation_weight=4.0,
            specificity_bonus=0.02,
            generation_penalty=penalty,
            max_candidates=base.max_candidates,
        )
        for penalty in (0.0, 0.05, 0.1, 0.2)
    )
    anchors.append(
        MemoryConfig(
            validation_weight=4.0,
            complexity_penalty=0.0,
            confidence_weight=0.0,
            frequency_weight=0.0,
            revision_bonus=0.0,
            generalization_bonus=0.0,
            formation_weight=4.0,
            specificity_bonus=0.0,
            max_candidates=base.max_candidates,
        )
    )
    for trial in range(len(anchors) + trials):
        if trial < len(anchors):
            sampled = anchors[trial]
        else:
            sampled = MemoryConfig(
                validation_weight=float(np.exp(rng.uniform(np.log(1.0), np.log(8.0)))),
                complexity_penalty=float(rng.uniform(0.0, 0.4)),
                confidence_weight=float(rng.uniform(0.0, 1.0)),
                frequency_weight=float(rng.uniform(0.0, 1.0)),
                revision_bonus=float(rng.uniform(0.0, 1.0)),
                generalization_bonus=float(rng.uniform(0.0, 0.5)),
                formation_weight=(
                    0.0 if rng.random() < 0.25 else float(np.exp(rng.uniform(np.log(0.5), np.log(8.0))))
                ),
                specificity_bonus=0.0 if rng.random() < 0.25 else float(rng.uniform(0.0, 0.05)),
                generation_penalty=(
                    0.0 if rng.random() < 0.25 else float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
                ),
                max_candidates=base.max_candidates,
            )
        results = evaluate_trajectories(
            trajectories,
            {method: sampled for method in methods},
            library,
        )
        if compositional_trajectories is not None:
            compositional_results = evaluate_trajectories(
                compositional_trajectories,
                {method: sampled for method in methods},
                library,
            )
            results = pd.concat([results, compositional_results], ignore_index=True)
        for method in methods:
            score = tuning_objective(results, method)
            history.append({"trial": trial, "method": method, "score": score, **asdict(sampled)})
            if score > best_scores[method]:
                best_scores[method] = score
                best_configs[method] = sampled
    return best_configs, pd.DataFrame(history)
