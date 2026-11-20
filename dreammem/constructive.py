from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MemoryConfig
from .diagnostics import constructive_closure, state_candidates
from .evaluation import decision_metrics
from .memory import MemoryDecision, MemoryFormer, _error, _softmax_scores
from .tasks import (
    FAMILY_VALUES,
    N_ATTRIBUTES,
    ConceptLibrary,
    attribute_family,
    code_to_concept,
    concept_label,
    concept_to_code,
    valid_concept,
)
from .trajectory import EpisodeTrajectory


SEARCH_METHODS = ("closure_search", "random_search", "learned_search")


def evidence_decision(
    trajectory: EpisodeTrajectory,
    candidates: set[int],
    config: MemoryConfig,
    method: str,
    library: ConceptLibrary | None = None,
) -> MemoryDecision:
    library = library or ConceptLibrary()
    if not candidates:
        return MemoryDecision(method, None, {}, 0, 0.5, False)
    occurrence = MemoryFormer(config, library)._occurrences(trajectory)
    counts = occurrence["counts"]
    confidence_sum = occurrence["confidence_sum"]
    final_codes = occurrence["final_codes"]
    validation_start = max(stage.support_count for stage in trajectory.stages)
    scores = {}
    validation_errors = {}
    for code in candidates:
        validation_error = _error(code, trajectory.episode, validation_start, library=library)
        formation_error = _error(code, trajectory.episode, 0, validation_start, library)
        concept = library.code_to_concept(code)
        score = (
            -config.validation_weight * validation_error
            -config.formation_weight * formation_error
            -config.complexity_penalty * len(concept)
            +config.specificity_bonus * len(concept)
        )
        if counts[code] == 0:
            score -= config.generation_penalty
        scores[code] = score
        validation_errors[code] = validation_error
    posterior = _softmax_scores(scores, config.max_candidates)
    best = max(posterior, key=posterior.get)
    return MemoryDecision(
        method=method,
        concept=library.code_to_concept(best),
        posterior=posterior,
        candidate_count=len(candidates),
        validation_error=validation_errors[best],
        novel_candidate=best not in final_codes,
        library=library,
    )


def candidate_features(
    code: int,
    trajectory: EpisodeTrajectory,
    occurrence: dict | None = None,
    generated: set[int] | None = None,
    feature_mode: str = "full",
    library: ConceptLibrary | None = None,
) -> np.ndarray:
    library = library or ConceptLibrary()
    episode = trajectory.episode
    concept = library.code_to_concept(code)
    raw_errors = np.asarray(
        [float(concept_label(concept, example.attributes) != example.label) for example in episode.support],
        dtype=np.float32,
    )
    positives = np.asarray([example.label for example in episode.support], dtype=bool)
    negative_error = float(raw_errors[~positives].mean()) if (~positives).any() else 0.5
    positive_error = float(raw_errors[positives].mean()) if positives.any() else 0.5
    formation_count = max(stage.support_count for stage in trajectory.stages)
    formation_error = float(raw_errors[:formation_count].mean())
    validation_error = (
        float(raw_errors[formation_count:].mean()) if len(raw_errors) > formation_count else 0.5
    )
    errors = np.full(8, 0.5, dtype=np.float32)
    errors[: min(8, len(raw_errors))] = raw_errors[:8]
    occurrence = occurrence or MemoryFormer(MemoryConfig(), library)._occurrences(trajectory)
    counts = occurrence["counts"]
    confidence_sum = occurrence["confidence_sum"]
    total = max(int(occurrence["total"]), 1)
    attributes = np.zeros(library.n_attributes, dtype=np.float32)
    attributes[list(concept)] = 1.0
    families = np.zeros(len(library.families), dtype=np.float32)
    families[[library.attribute_family(value) for value in concept]] = 1.0
    sizes = np.zeros(3, dtype=np.float32)
    sizes[len(concept) - 1] = 1.0
    features = np.concatenate(
        [
            errors,
            np.asarray(
                [
                    formation_error,
                    validation_error,
                    positive_error,
                    negative_error,
                    len(concept) / 3.0,
                ],
                dtype=np.float32,
            ),
            sizes,
            attributes,
            families,
            np.asarray(
                [
                    counts[code] / total,
                    confidence_sum[code] / max(counts[code], 1),
                    float(code in occurrence["final_codes"]),
                    float(code in (generated or set())),
                ],
                dtype=np.float32,
            ),
        ]
    )
    features = features.astype(np.float32)
    if feature_mode == "full":
        return features
    if feature_mode == "no_evidence":
        return features[12:]
    if feature_mode == "no_trajectory":
        return features[:33]
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def concept_neighbors(
    code: int,
    pair_pool: set[int],
    library: ConceptLibrary | None = None,
) -> set[int]:
    library = library or ConceptLibrary()
    concept = library.code_to_concept(code)
    neighbors = set()
    if len(concept) > 1:
        for removed in concept:
            neighbors.add(concept_to_code(tuple(value for value in concept if value != removed)))
    for value in range(library.n_attributes):
        if value in concept:
            continue
        expanded = tuple(sorted((*concept, value)))
        if library.valid_concept(expanded):
            neighbors.add(concept_to_code(expanded))
        for removed in concept:
            substituted = tuple(sorted((*[item for item in concept if item != removed], value)))
            if library.valid_concept(substituted):
                neighbors.add(concept_to_code(substituted))
    for other in pair_pool:
        if other == code:
            continue
        left = set(concept)
        right = set(library.code_to_concept(other))
        union = tuple(sorted(left | right))
        intersection = tuple(sorted(left & right))
        if library.valid_concept(union):
            neighbors.add(concept_to_code(union))
        if intersection:
            neighbors.add(concept_to_code(intersection))
    neighbors.discard(code)
    return neighbors


@dataclass
class SearchPolicy:
    estimator: object
    branch: int = 4
    pair_pool: int = 8
    feature_mode: str = "full"
    library: ConceptLibrary | None = None

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(features)[:, 1]


def fit_search_policy(
    trajectories: list[EpisodeTrajectory],
    library: ConceptLibrary,
    seed: int,
    branch: int = 4,
    pair_pool: int = 8,
    feature_mode: str = "full",
    estimator_kind: str = "hgb",
) -> SearchPolicy:
    features = []
    labels = []
    for trajectory in trajectories:
        former = MemoryFormer(MemoryConfig(), library)
        occurrence = former._occurrences(trajectory)
        generated = former._revision_scores(trajectory)[1]
        true_code = concept_to_code(trajectory.episode.concept)
        concepts = library.all_concepts(max_size=3)
        features.extend(
            candidate_features(
                concept_to_code(concept), trajectory, occurrence, generated, feature_mode, library
            )
            for concept in concepts
        )
        labels.extend(float(concept_to_code(concept) == true_code) for concept in concepts)
    features = np.asarray(features)
    labels = np.asarray(labels)
    if estimator_kind == "hgb":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            class_weight="balanced",
            random_state=seed,
        )
        estimator.fit(features, labels)
    elif estimator_kind == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        )
        estimator.fit(features, labels)
    elif estimator_kind == "mlp":
        estimator = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            ),
        )
        positive = max(float(labels.sum()), 1.0)
        weights = np.where(labels > 0, (len(labels) - positive) / positive, 1.0)
        estimator.fit(features, labels, mlpclassifier__sample_weight=weights)
    else:
        raise ValueError(estimator_kind)
    return SearchPolicy(estimator, branch, pair_pool, feature_mode, library)


def initial_candidates(
    trajectory: EpisodeTrajectory,
    library: ConceptLibrary | None = None,
) -> set[int]:
    former = MemoryFormer(MemoryConfig(), library)
    candidates = set(former._occurrences(trajectory)["counts"])
    candidates |= former._revision_scores(trajectory)[1]
    return candidates


def search_candidates(
    trajectory: EpisodeTrajectory,
    policy: SearchPolicy | None,
    budget: int,
    strategy: str,
    seed: int,
    initial_codes: set[int] | None = None,
    allow_merge: bool = True,
    library: ConceptLibrary | None = None,
) -> tuple[set[int], dict[str, float]]:
    library = library or ConceptLibrary()
    initial = initial_candidates(trajectory, library) if initial_codes is None else set(initial_codes)
    occurrence = MemoryFormer(MemoryConfig(), library)._occurrences(trajectory)
    generated = MemoryFormer(MemoryConfig(), library)._revision_scores(trajectory)[1]
    rng = np.random.default_rng(seed)
    expanded = set()
    visited = set(initial)
    generated_unique = set()
    generated_instances = 0
    controller_score_calls = 0
    controller_scored_instances = 0
    policy_scored_instances = 0
    controller_scored_unique = set()

    def scores(candidates: set[int]) -> dict[int, float]:
        nonlocal controller_score_calls, controller_scored_instances, policy_scored_instances
        if not candidates:
            return {}
        ordered = sorted(candidates)
        controller_score_calls += 1
        controller_scored_instances += len(ordered)
        controller_scored_unique.update(ordered)
        if strategy.startswith("learned_search"):
            features = np.asarray(
                [
                    candidate_features(code, trajectory, occurrence, generated, policy.feature_mode, library)
                    for code in ordered
                ]
            )
            values = policy.scores(features)
            policy_scored_instances += len(ordered)
        else:
            values = rng.random(len(ordered))
        return dict(zip(ordered, values))

    if len(visited) > budget:
        ranked = scores(visited)
        visited = set(sorted(visited, key=ranked.get, reverse=True)[:budget])
    expansions = 0
    while len(visited) < budget:
        remaining = visited - expanded
        if not remaining:
            break
        ranked = scores(visited)
        parents = sorted(remaining, key=ranked.get, reverse=True)[: policy.branch if policy else 4]
        pair_pool = (
            set(sorted(visited, key=ranked.get, reverse=True)[: policy.pair_pool if policy else 8])
            if allow_merge
            else set()
        )
        new = set()
        for parent in parents:
            neighbors = concept_neighbors(parent, pair_pool, library)
            generated_unique.update(neighbors)
            generated_instances += len(neighbors)
            new |= neighbors
            expanded.add(parent)
            expansions += 1
        new -= visited
        available = budget - len(visited)
        if not new or available <= 0:
            break
        new_scores = scores(new)
        visited |= set(sorted(new, key=new_scores.get, reverse=True)[:available])
    true_code = concept_to_code(trajectory.episode.concept)
    return visited, {
        "candidate_recall": float(true_code in visited),
        "initial_candidate_count": len(initial),
        "expansions": expansions,
        "search_budget": budget,
        "generated_unique_candidates": len(generated_unique),
        "generated_candidate_instances": generated_instances,
        "controller_score_calls": controller_score_calls,
        "controller_scored_instances": controller_scored_instances,
        "controller_scored_unique": len(controller_scored_unique),
        "policy_scored_instances": policy_scored_instances,
        "evidence_scored_candidates": len(visited),
    }


def evaluate_search(
    trajectories: list[EpisodeTrajectory],
    config: MemoryConfig,
    library: ConceptLibrary,
    policy: SearchPolicy | None,
    budget: int,
    strategy: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for trajectory in trajectories:
        started = time.perf_counter()
        if strategy == "closure_search":
            statistics = {}
            candidates = constructive_closure(state_candidates(trajectory), statistics, library)
            information = {
                "candidate_recall": float(concept_to_code(trajectory.episode.concept) in candidates),
                "initial_candidate_count": len(state_candidates(trajectory)),
                "expansions": 0,
                "search_budget": len(candidates),
                "controller_score_calls": 0,
                "controller_scored_instances": 0,
                "controller_scored_unique": 0,
                "policy_scored_instances": 0,
                "evidence_scored_candidates": len(candidates),
                **statistics,
            }
        else:
            initial_codes = (
                {concept_to_code(concept) for concept in library.all_concepts(max_size=1)}
                if strategy.endswith("_atoms")
                else None
            )
            candidates, information = search_candidates(
                trajectory,
                policy,
                budget,
                strategy,
                seed + trajectory.episode.index,
                initial_codes=initial_codes,
                allow_merge=not strategy.endswith("_no_merge"),
                library=library,
            )
        search_seconds = time.perf_counter() - started
        evidence_started = time.perf_counter()
        decision = evidence_decision(trajectory, candidates, config, strategy, library)
        evidence_seconds = time.perf_counter() - evidence_started
        rows.append(decision_metrics(decision, trajectory))
        diagnostics.append(
            {
                "split": trajectory.episode.split,
                "episode": trajectory.episode.index,
                "method": strategy,
                "candidate_count": len(candidates),
                "search_seconds": search_seconds,
                "evidence_seconds": evidence_seconds,
                **information,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def policy_decision(
    trajectory: EpisodeTrajectory,
    candidates: set[int],
    policy: SearchPolicy,
    config: MemoryConfig,
    method: str,
) -> MemoryDecision:
    library = policy.library or ConceptLibrary()
    if not candidates:
        return MemoryDecision(method, None, {}, 0, 0.5, False, library)
    ordered = sorted(candidates)
    occurrence = MemoryFormer(config, library)._occurrences(trajectory)
    generated = MemoryFormer(config, library)._revision_scores(trajectory)[1]
    features = np.asarray(
        [
            candidate_features(code, trajectory, occurrence, generated, policy.feature_mode, library)
            for code in ordered
        ]
    )
    values = policy.scores(features)
    selected_indices = np.argsort(-values, kind="stable")[: config.max_candidates]
    selected = [ordered[index] for index in selected_indices]
    weights = {code: float(values[index]) for code, index in zip(selected, selected_indices)}
    normalizer = sum(weights.values())
    if normalizer <= 0:
        posterior = {code: 1.0 / len(selected) for code in selected}
    else:
        posterior = {code: weight / normalizer for code, weight in weights.items()}
    best = max(posterior, key=posterior.get)
    validation_start = max(stage.support_count for stage in trajectory.stages)
    return MemoryDecision(
        method=method,
        concept=library.code_to_concept(best),
        posterior=posterior,
        candidate_count=len(candidates),
        validation_error=_error(best, trajectory.episode, validation_start, library=library),
        novel_candidate=best not in occurrence["final_codes"],
        library=library,
    )


def evaluate_policy_ranker(
    trajectories: list[EpisodeTrajectory],
    config: MemoryConfig,
    library: ConceptLibrary,
    policy: SearchPolicy,
    budget: int,
    strategy: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for trajectory in trajectories:
        started = time.perf_counter()
        if strategy == "learned_ranker_full":
            candidates = {
                concept_to_code(concept) for concept in library.concepts_for_split(trajectory.episode.split)
            }
            information = {
                "generated_unique_candidates": 0,
                "generated_candidate_instances": 0,
                "expansions": 0,
            }
        elif strategy == "learned_ranker_initial":
            candidates = initial_candidates(trajectory, library)
            information = {
                "generated_unique_candidates": 0,
                "generated_candidate_instances": 0,
                "expansions": 0,
            }
        elif strategy == "learned_search_policy_rank":
            candidates, information = search_candidates(
                trajectory,
                policy,
                budget,
                "learned_search",
                seed + trajectory.episode.index,
                library=library,
            )
        else:
            raise ValueError(strategy)
        decision = policy_decision(trajectory, candidates, policy, config, strategy)
        rows.append(decision_metrics(decision, trajectory))
        diagnostics.append(
            {
                "split": trajectory.episode.split,
                "episode": trajectory.episode.index,
                "method": strategy,
                **information,
                "candidate_recall": float(concept_to_code(trajectory.episode.concept) in candidates),
                "candidate_count": len(candidates),
                "initial_candidate_count": len(initial_candidates(trajectory, library)),
                "search_budget": budget if strategy == "learned_search_policy_rank" else len(candidates),
                "controller_score_calls": information.get("controller_score_calls", 0) + 1,
                "controller_scored_instances": information.get("controller_scored_instances", 0) + len(candidates),
                "controller_scored_unique": information.get("controller_scored_unique", len(candidates)),
                "policy_scored_instances": information.get("policy_scored_instances", 0) + len(candidates),
                "ranker_scored_candidates": len(candidates),
                "evidence_scored_candidates": 0,
                "search_seconds": time.perf_counter() - started,
                "evidence_seconds": 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)
