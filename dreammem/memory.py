from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import exp

import numpy as np

from .config import MemoryConfig
from .encoding import ATTRIBUTE_BASE, NIL
from .semantics import INVALID_CONCEPT, RevisionType, revision_type
from .tasks import (
    Episode,
    ConceptLibrary,
    N_ATTRIBUTES,
    code_to_concept,
    concept_label,
    concept_to_code,
    valid_concept,
)
from .trajectory import EpisodeTrajectory


MEMORY_METHODS = (
    "final",
    "token_stability",
    "semantic_stability",
    "bag_of_states",
    "revision_no_merge",
    "merge_only",
    "revision_dynamics",
    "evidence_states",
    "evidence_guided",
)

_TRAJECTORY_CACHE = {}


def _cache_trajectory_method(function):
    name = function.__name__

    def wrapper(self, trajectory):
        key = (name, id(trajectory))
        if key not in _TRAJECTORY_CACHE:
            _TRAJECTORY_CACHE[key] = (trajectory, function(self, trajectory))
        return _TRAJECTORY_CACHE[key][1]

    return wrapper


@dataclass
class MemoryDecision:
    method: str
    concept: tuple[int, ...] | None
    posterior: dict[int, float]
    candidate_count: int
    validation_error: float
    novel_candidate: bool
    library: ConceptLibrary | None = None

    def query_probability(self, attributes: tuple[int, ...]) -> float:
        if not self.posterior:
            return 0.5
        library = self.library or ConceptLibrary()
        return sum(
            probability
            for code, probability in self.posterior.items()
            if concept_label(library.code_to_concept(code), attributes)
        )


def _error(
    concept_code: int,
    episode: Episode,
    start: int = 0,
    stop: int | None = None,
    library: ConceptLibrary | None = None,
) -> float:
    examples = episode.support[start:stop]
    if not examples:
        return 0.5
    concept = (library or ConceptLibrary()).code_to_concept(concept_code)
    return float(
        np.mean([concept_label(concept, example.attributes) != example.label for example in examples])
    )


def _concept_slots(code: int, library: ConceptLibrary | None = None) -> tuple[int, int, int]:
    attributes = list((library or ConceptLibrary()).code_to_concept(code))
    values = [ATTRIBUTE_BASE + value for value in attributes]
    values.extend([NIL] * (3 - len(values)))
    return tuple(values)


def _softmax_scores(scores: dict[int, float], limit: int) -> dict[int, float]:
    if not scores:
        return {}
    selected = sorted(scores, key=scores.get, reverse=True)[:limit]
    maximum = max(scores[code] for code in selected)
    weights = {code: exp(scores[code] - maximum) for code in selected}
    normalizer = sum(weights.values())
    return {code: weight / normalizer for code, weight in weights.items()}


class MemoryFormer:
    def __init__(self, config: MemoryConfig, library: ConceptLibrary | None = None) -> None:
        self.config = config
        self.library = library or ConceptLibrary()

    @_cache_trajectory_method
    def _occurrences(self, trajectory: EpisodeTrajectory) -> dict[str, object]:
        counts: Counter[int] = Counter()
        final_counts: Counter[int] = Counter()
        confidence_sum: defaultdict[int, float] = defaultdict(float)
        final_confidence_sum: defaultdict[int, float] = defaultdict(float)
        final_codes: set[int] = set()
        total = 0
        for stage in trajectory.stages:
            codes = stage.concept_codes
            confidence = stage.concept_confidence.astype(np.float32).mean(axis=-1)
            for sample in range(codes.shape[0]):
                for step in range(codes.shape[1]):
                    code = int(codes[sample, step])
                    if code == INVALID_CONCEPT:
                        continue
                    counts[code] += 1
                    confidence_sum[code] += float(confidence[sample, step])
                    total += 1
                final_code = int(codes[sample, -1])
                if final_code != INVALID_CONCEPT:
                    final_counts[final_code] += 1
                    final_confidence_sum[final_code] += float(confidence[sample, -1])
                    final_codes.add(final_code)
        return {
            "counts": counts,
            "final_counts": final_counts,
            "confidence_sum": confidence_sum,
            "final_confidence_sum": final_confidence_sum,
            "final_codes": final_codes,
            "total": total,
        }

    @_cache_trajectory_method
    def _token_stability(self, trajectory: EpisodeTrajectory) -> list[dict[int, float]]:
        probabilities = []
        for position in range(3):
            counter: Counter[int] = Counter()
            for stage in trajectory.stages:
                counter.update(stage.concept_tokens[:, :, position].reshape(-1).tolist())
            total = sum(counter.values())
            probabilities.append({token: count / total for token, count in counter.items()})
        return probabilities

    @_cache_trajectory_method
    def _semantic_stability(self, trajectory: EpisodeTrajectory) -> defaultdict[int, float]:
        scores: defaultdict[int, float] = defaultdict(float)
        trajectories = 0
        for stage in trajectory.stages:
            for sample in range(stage.concept_codes.shape[0]):
                codes = stage.concept_codes[sample]
                trajectories += 1
                runs: defaultdict[int, list[int]] = defaultdict(list)
                previous = INVALID_CONCEPT
                length = 0
                for raw in list(codes) + [INVALID_CONCEPT]:
                    code = int(raw)
                    if code == previous:
                        length += 1
                    else:
                        if previous != INVALID_CONCEPT:
                            runs[previous].append(length)
                        previous = code
                        length = 1
                for code, lengths in runs.items():
                    scores[code] += max(lengths) / len(codes)
        if trajectories:
            for code in scores:
                scores[code] /= trajectories
        return scores

    @_cache_trajectory_method
    def _revision_raw(
        self,
        trajectory: EpisodeTrajectory,
    ) -> tuple[defaultdict[int, float], defaultdict[int, float], set[int]]:
        values: defaultdict[int, float] = defaultdict(float)
        generalizations: defaultdict[int, float] = defaultdict(float)
        visits: Counter[int] = Counter()
        generated: set[int] = set()

        def record(before: int, after: int, support_count: int) -> None:
            if before == INVALID_CONCEPT or after == INVALID_CONCEPT or before == after:
                return
            improvement = _error(before, trajectory.episode, 0, support_count, self.library) - _error(
                after, trajectory.episode, 0, support_count, self.library
            )
            kind = revision_type(before, after, self.library)
            values[after] += improvement
            generalizations[after] += float(kind == RevisionType.GENERALIZE)
            visits[after] += 1
            previous = set(self.library.code_to_concept(before))
            current = set(self.library.code_to_concept(after))
            union = tuple(sorted(previous | current))
            intersection = tuple(sorted(previous & current))
            if self.library.valid_concept(union):
                generated.add(concept_to_code(union))
            if intersection:
                generated.add(concept_to_code(intersection))

        for stage in trajectory.stages:
            codes = stage.concept_codes
            for sample in range(codes.shape[0]):
                for step in range(codes.shape[1] - 1):
                    record(
                        int(codes[sample, step]),
                        int(codes[sample, step + 1]),
                        stage.support_count,
                    )
        for previous_stage, current_stage in zip(trajectory.stages, trajectory.stages[1:]):
            for sample in range(previous_stage.concept_codes.shape[0]):
                before = int(previous_stage.concept_codes[sample, -1])
                for after in current_stage.concept_codes[sample]:
                    record(before, int(after), current_stage.support_count)
        for code in values:
            values[code] /= visits[code]
            generalizations[code] /= visits[code]
        return values, generalizations, generated

    def _revision_scores(
        self,
        trajectory: EpisodeTrajectory,
    ) -> tuple[defaultdict[int, float], set[int]]:
        improvements, generalizations, generated = self._revision_raw(trajectory)
        values = defaultdict(float)
        for code, improvement in improvements.items():
            values[code] = improvement + self.config.generalization_bonus * generalizations[code]
        return values, generated

    def form(self, trajectory: EpisodeTrajectory, method: str) -> MemoryDecision:
        if method not in MEMORY_METHODS:
            raise ValueError(f"Unknown memory method: {method}")
        occurrence = self._occurrences(trajectory)
        counts: Counter[int] = occurrence["counts"]
        final_counts: Counter[int] = occurrence["final_counts"]
        confidence_sum: defaultdict[int, float] = occurrence["confidence_sum"]
        final_confidence_sum: defaultdict[int, float] = occurrence["final_confidence_sum"]
        final_codes: set[int] = occurrence["final_codes"]
        total = int(occurrence["total"])
        candidates = set(final_counts) if method == "final" else set(counts)
        revision_scores: defaultdict[int, float] = defaultdict(float)
        if method in ("revision_dynamics", "revision_no_merge", "merge_only", "evidence_guided"):
            revision_scores, generated = self._revision_scores(trajectory)
            if method != "revision_no_merge":
                candidates |= generated
        if not candidates:
            return MemoryDecision(method, None, {}, 0, 0.5, False)
        token_probabilities = self._token_stability(trajectory) if method == "token_stability" else []
        semantic_scores = self._semantic_stability(trajectory) if method == "semantic_stability" else {}
        validation_start = max(stage.support_count for stage in trajectory.stages)
        scores: dict[int, float] = {}
        validation_errors: dict[int, float] = {}
        for code in candidates:
            validation_error = _error(code, trajectory.episode, validation_start, library=self.library)
            validation_errors[code] = validation_error
            concept = self.library.code_to_concept(code)
            frequency = counts[code] / max(total, 1)
            if method == "final":
                confidence = final_confidence_sum[code] / max(final_counts[code], 1)
            else:
                confidence = confidence_sum[code] / max(counts[code], 1)
            formation_error = (
                _error(code, trajectory.episode, 0, validation_start, self.library)
                if self.config.formation_weight
                else 0.0
            )
            score = (
                -self.config.validation_weight * validation_error
                -self.config.formation_weight * formation_error
                -self.config.complexity_penalty * len(concept)
                +self.config.specificity_bonus * len(concept)
                +(
                    0.0
                    if method in ("evidence_states", "evidence_guided")
                    else self.config.confidence_weight * confidence
                )
            )
            if counts[code] == 0 and method in ("merge_only", "revision_dynamics", "evidence_guided"):
                score -= self.config.generation_penalty
            if method == "final":
                score += self.config.frequency_weight * final_counts[code] / max(
                    sum(final_counts.values()), 1
                )
            elif method == "token_stability":
                slots = _concept_slots(code, self.library)
                score += float(
                    np.mean([token_probabilities[position].get(token, 0.0) for position, token in enumerate(slots)])
                )
            elif method == "semantic_stability":
                score += semantic_scores.get(code, 0.0)
            elif method in ("bag_of_states", "merge_only"):
                score += self.config.frequency_weight * frequency
            elif method in ("revision_dynamics", "revision_no_merge"):
                score += self.config.revision_bonus * revision_scores.get(code, 0.0)
                score += self.config.frequency_weight * frequency
            scores[code] = score
        posterior = _softmax_scores(scores, self.config.max_candidates)
        best = max(posterior, key=posterior.get)
        return MemoryDecision(
            method=method,
            concept=self.library.code_to_concept(best),
            posterior=posterior,
            candidate_count=len(candidates),
            validation_error=validation_errors[best],
            novel_candidate=best not in final_codes,
            library=self.library,
        )


@lru_cache(maxsize=2048)
def _symbolic_errors(
    episode: Episode,
    concepts: tuple[tuple[int, ...], ...],
    library: ConceptLibrary | None = None,
) -> dict[int, float]:
    return {
        concept_to_code(concept): _error(concept_to_code(concept), episode, library=library)
        for concept in concepts
    }


def symbolic_memory(
    episode: Episode,
    concepts: tuple[tuple[int, ...], ...],
    config: MemoryConfig,
    library: ConceptLibrary | None = None,
) -> MemoryDecision:
    library = library or ConceptLibrary()
    scores = {}
    errors = _symbolic_errors(episode, concepts, library)
    for concept in concepts:
        code = concept_to_code(concept)
        error = errors[code]
        scores[code] = (
            -config.validation_weight * error
            -config.complexity_penalty * len(concept)
            +config.specificity_bonus * len(concept)
        )
    posterior = _softmax_scores(scores, config.max_candidates)
    best = max(posterior, key=posterior.get)
    return MemoryDecision(
        method="symbolic",
        concept=library.code_to_concept(best),
        posterior=posterior,
        candidate_count=len(concepts),
        validation_error=errors[best],
        novel_candidate=False,
        library=library,
    )
