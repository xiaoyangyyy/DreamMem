from __future__ import annotations

from itertools import product

import numpy as np

from .config import DataConfig
from .operators import OperatorConceptLibrary, OperatorTaskGenerator, OperatorTextEncoder
from .tasks import Example, attribute_family, concept_label


class AdvancedConceptLibrary(OperatorConceptLibrary):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.by_split["threshold"] = tuple(concept for concept in self.concepts if 12 in concept)
        self.by_split["transitive"] = tuple(concept for concept in self.concepts if 12 in concept)
        self.split_concepts = {}

    def concepts_for_split(self, split: str):
        return self.split_concepts.get(split, self.all_concepts(max_size=3))


class AdvancedTaskGenerator(OperatorTaskGenerator):
    def __init__(self, config: DataConfig) -> None:
        super().__init__(config)
        self.library = AdvancedConceptLibrary(config.seed)
        for split, concepts in self.library.by_split.items():
            if split == "transitive":
                continue
            realizable = []
            for concept in concepts:
                try:
                    self._binary_possibilities(concept)
                    realizable.append(concept)
                except ValueError:
                    pass
            self.library.by_split[split] = tuple(realizable)
            self.library.split_concepts[split] = tuple(realizable)
        self._filter_transitive_concepts()

    def _filter_transitive_concepts(self) -> None:
        concepts = self.library.by_split["transitive"]
        positive = set()
        negative = set()
        objects = tuple(product(range(4), range(4), (8, 9), (10, 11, 12)))
        for left, right, third in product(objects, repeat=3):
            attributes = self._transitive_attributes(left, right, third)
            for concept in concepts:
                if concept in positive and concept in negative:
                    continue
                if concept_label(concept, attributes):
                    positive.add(concept)
                else:
                    negative.add(concept)
            if len(positive & negative) == len(concepts):
                break
        realizable = tuple(concept for concept in concepts if concept in positive and concept in negative)
        self.library.by_split["transitive"] = realizable
        self.library.split_concepts["transitive"] = realizable

    def _attributes(self, left, right):
        attributes = [
            0 if left[0] == right[0] else 1,
            4 if left[1] == right[1] else 5,
            9 if left[2] > right[2] else 8,
            10 if left[3] == right[3] else 11,
        ]
        if left[0] != 0:
            attributes.append(2)
        if left[0] == 0 or right[0] == 0:
            attributes.append(3)
        if (left[1] == 4) != (right[1] == 4):
            attributes.append(6)
        if left[1] != 4 or right[1] == 4:
            attributes.append(7)
        if left[2] + right[2] >= 17:
            attributes.append(12)
        return tuple(attributes)

    def _link(self, left: int, right: int) -> bool:
        return (left * 7 + right * 11 + self.config.seed) % 13 < 5

    def _transitive_attributes(self, left, right, third):
        attributes = list(self._attributes(left, right))
        if self._link(left[1], right[1]) and self._link(right[1], third[1]) and self._link(left[1], third[1]):
            attributes.append(12)
        return tuple(attributes)

    def _transitive_example(self, rng, concept, positive):
        if positive is None:
            positive = bool(rng.integers(0, 2))
        if not positive and rng.random() < self.config.near_miss_probability:
            base = self._transitive_example(rng, concept, True)
            required = int(rng.choice(concept))
            family = attribute_family(required)
            for _ in range(64):
                left = list(base.left)
                right = list(base.right)
                third = list(base.third)
                target = (left, right, third)[int(rng.integers(3))]
                values = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12))[family]
                target[family] = int(rng.choice(values))
                attributes = self._transitive_attributes(tuple(left), tuple(right), tuple(third))
                if required not in attributes:
                    return Example(attributes, False, tuple(left), tuple(right), tuple(third))
        for _ in range(10000):
            left = tuple(int(rng.choice(values)) for values in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12)))
            right = tuple(int(rng.choice(values)) for values in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12)))
            third = tuple(int(rng.choice(values)) for values in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12)))
            attributes = self._transitive_attributes(left, right, third)
            if concept_label(concept, attributes) == positive:
                return Example(attributes, positive, left, right, third)
        objects = tuple(product(range(4), range(4), (8, 9), (10, 11, 12)))
        for left, right, third in product(objects, repeat=3):
            attributes = self._transitive_attributes(left, right, third)
            if concept_label(concept, attributes) == positive:
                return Example(attributes, positive, left, right, third)
        raise ValueError(f"No valid transitive example for {concept}")

    def _threshold_example(self, rng, concept, positive):
        if positive is None:
            positive = bool(rng.integers(0, 2))
        if not positive and rng.random() < self.config.near_miss_probability:
            base = self._threshold_example(rng, concept, True)
            required = int(rng.choice(concept))
            family = 2 if required == 12 else attribute_family(required)
            for _ in range(64):
                left = list(base.left)
                right = list(base.right)
                target = left if rng.random() < 0.5 else right
                values = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12))[family]
                target[family] = int(rng.choice(values))
                attributes = self._attributes(tuple(left), tuple(right))
                if required not in attributes:
                    return Example(attributes, False, tuple(left), tuple(right))
            values = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12))[family]
            while True:
                left = list(base.left)
                right = list(base.right)
                left[family] = int(rng.choice(values))
                right[family] = int(rng.choice(values))
                attributes = self._attributes(tuple(left), tuple(right))
                if required not in attributes:
                    return Example(attributes, False, tuple(left), tuple(right))
        while True:
            left = tuple(int(rng.choice(values)) for values in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12)))
            right = tuple(int(rng.choice(values)) for values in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11, 12)))
            attributes = self._attributes(left, right)
            if concept_label(concept, attributes) == positive:
                return Example(attributes, positive, left, right)

    def _example(self, rng, concept, positive):
        return self._transitive_example(rng, concept, positive)

    def sample(self, split: str, index: int, seed: int):
        if split == "transitive":
            self._example = self._transitive_example
        else:
            self._example = super()._example
        return super().sample(split, index, seed)


class AdvancedTextEncoder(OperatorTextEncoder):
    def __init__(self, tokenizer, config: DataConfig, mask_token_id: int) -> None:
        super().__init__(tokenizer, config, mask_token_id)
        tokens = list(self.concept_attribute_tokens)
        tokens[12] = self._single_token("above")
        self.concept_attribute_tokens = tuple(tokens)
        self.attribute_by_token = {
            self.concept_attribute_tokens[attribute]: attribute for attribute in range(13)
        }
        self._template = 0

    def _example_text(self, example: Example) -> str:
        text = super()._example_text(example)
        if example.third is not None:
            text += f" third={self._object(example.third)}"
        return text
