from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from .config import DataConfig
from .relational import RelationalConceptLibrary, RelationalTaskGenerator, RelationalTextEncoder
from .tasks import ConceptLibrary, Example, attribute_family, concept_label


NEGATION_CODE = 2
DISJUNCTION_CODE = 3
XOR_CODES = (6, 12)
IMPLICATION_CODE = 7


def operator_attributes(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left_red = left[0] == 0
    right_red = right[0] == 0
    left_circle = left[1] == 4
    right_circle = right[1] == 4
    left_plain = left[3] == 10
    right_plain = right[3] == 10
    attributes = [
        0 if left[0] == right[0] else 1,
        4 if left[1] == right[1] else 5,
        9 if left[2] > right[2] else 8,
        10 if left[3] == right[3] else 11,
    ]
    if not left_red:
        attributes.append(NEGATION_CODE)
    if left_red or right_red:
        attributes.append(DISJUNCTION_CODE)
    if left_circle != right_circle:
        attributes.append(6)
    if not left_circle or right_circle:
        attributes.append(IMPLICATION_CODE)
    if left_plain != right_plain:
        attributes.append(12)
    return tuple(attributes)


class OperatorConceptLibrary(RelationalConceptLibrary):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        direct = RelationalConceptLibrary(seed)
        self.concepts = ConceptLibrary.all_concepts(self, max_size=3)
        self.by_split = dict(direct.by_split)
        self.by_split["negation"] = tuple(
            concept for concept in self.concepts if NEGATION_CODE in concept
        )
        self.by_split["disjunction"] = tuple(
            concept for concept in self.concepts if DISJUNCTION_CODE in concept
        )
        self.by_split["xor"] = tuple(
            concept for concept in self.concepts if any(code in concept for code in XOR_CODES)
        )
        self.by_split["implication"] = tuple(
            concept for concept in self.concepts if IMPLICATION_CODE in concept
        )


class OperatorTaskGenerator(RelationalTaskGenerator):
    def __init__(self, config: DataConfig) -> None:
        super().__init__(config)
        self.library = OperatorConceptLibrary(config.seed)

    def _attributes(self, left, right):
        return operator_attributes(left, right)

    @lru_cache(maxsize=256)
    def _binary_possibilities(self, concept):
        objects = tuple(product(range(4), range(4), (8, 9), (10, 11, 12)))
        positives = []
        negatives = []
        for left in objects:
            for right in objects:
                attributes = self._attributes(left, right)
                destination = positives if concept_label(concept, attributes) else negatives
                destination.append((left, right))
        if not positives or not negatives:
            raise ValueError(f"Concept has no valid positive or negative examples: {concept}")
        return np.asarray(positives), np.asarray(negatives)

    def _example(self, rng: np.random.Generator, concept, positive):
        if positive is None:
            positive = bool(rng.integers(0, 2))
        positives, negatives = self._binary_possibilities(concept)
        if positive:
            left, right = positives[int(rng.integers(len(positives)))]
            attributes = self._attributes(tuple(left), tuple(right))
            return Example(attributes, True, tuple(left), tuple(right))
        if rng.random() < self.config.near_miss_probability:
            base = positives[int(rng.integers(len(positives)))]
            distances = (negatives != base).sum(axis=(1, 2))
            nearest = np.flatnonzero(distances == distances.min())
            left, right = negatives[int(rng.choice(nearest))]
            attributes = self._attributes(tuple(left), tuple(right))
            return Example(attributes, False, tuple(left), tuple(right))
        left, right = negatives[int(rng.integers(len(negatives)))]
        attributes = self._attributes(tuple(left), tuple(right))
        return Example(attributes, False, tuple(left), tuple(right))


class OperatorTextEncoder(RelationalTextEncoder):
    def __init__(self, tokenizer, config: DataConfig, mask_token_id: int) -> None:
        super().__init__(tokenizer, config, mask_token_id)
        tokens = list(self.concept_attribute_tokens)
        tokens[NEGATION_CODE] = self._single_token("not")
        tokens[DISJUNCTION_CODE] = self._single_token("or")
        tokens[6] = self._single_token("xor")
        tokens[IMPLICATION_CODE] = self._single_token("if")
        tokens[12] = self._single_token("either")
        self.concept_attribute_tokens = tuple(tokens)
        self.attribute_by_token = {
            self.concept_attribute_tokens[attribute]: attribute for attribute in range(13)
        }
