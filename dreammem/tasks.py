from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from .config import DataConfig


FAMILY_VALUES: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9),
    (10, 11, 12),
)
N_ATTRIBUTES = 13
Split = Literal["train", "seen", "val", "unseen", "val_compositional", "compositional"]


@dataclass(frozen=True)
class Example:
    attributes: tuple[int, ...]
    label: bool
    left: tuple[int, int, int, int] | None = None
    right: tuple[int, int, int, int] | None = None
    third: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class Episode:
    concept: tuple[int, ...]
    support: tuple[Example, ...]
    queries: tuple[Example, ...]
    split: Split
    index: int
    template: int = 0


def attribute_family(attribute: int) -> int:
    for family, values in enumerate(FAMILY_VALUES):
        if attribute in values:
            return family
    raise ValueError(f"Unknown attribute: {attribute}")


def valid_concept(concept: tuple[int, ...]) -> bool:
    if not 1 <= len(concept) <= 3:
        return False
    if len(set(concept)) != len(concept):
        return False
    families = [attribute_family(value) for value in concept]
    return len(set(families)) == len(families)


def concept_label(concept: tuple[int, ...], attributes: tuple[int, ...]) -> bool:
    return all(value in attributes for value in concept)


def concept_to_code(concept: tuple[int, ...]) -> int:
    code = 0
    for attribute in concept:
        code |= 1 << attribute
    return code


def code_to_concept(code: int) -> tuple[int, ...]:
    return tuple(attribute for attribute in range(N_ATTRIBUTES) if code & (1 << attribute))


class ConceptLibrary:
    def __init__(self, seed: int = 123, families=None) -> None:
        self.families = tuple(families) if families is not None else FAMILY_VALUES
        self.n_attributes = sum(len(values) for values in self.families)
        self.atoms = tuple((value,) for values in self.families for value in values)
        pairs = [
            tuple(sorted(values))
            for values in combinations(range(self.n_attributes), 2)
            if self.valid_concept(tuple(sorted(values)))
        ]
        triples = [
            tuple(sorted(values))
            for values in combinations(range(self.n_attributes), 3)
            if self.valid_concept(tuple(sorted(values)))
        ]
        rng = np.random.default_rng(seed)
        rng.shuffle(pairs)
        rng.shuffle(triples)
        n_pairs = len(pairs)
        n_train = int(round(n_pairs * 0.8))
        n_val = int(round(n_pairs * 0.1))
        self.train_pairs = tuple(pairs[:n_train])
        self.val_pairs = tuple(pairs[n_train : n_train + n_val])
        self.test_pairs = tuple(pairs[n_train + n_val :])
        self.train = self.atoms + self.train_pairs
        self.seen = self.train
        self.val = self.val_pairs
        self.unseen = self.test_pairs
        self.val_compositional = tuple(triples[:16])
        self.compositional = tuple(triples[16:])
        self.by_split: dict[str, tuple[tuple[int, ...], ...]] = {
            "train": self.train,
            "seen": self.seen,
            "val": self.val,
            "unseen": self.unseen,
            "val_compositional": self.val_compositional,
            "compositional": self.compositional,
        }

    def attribute_family(self, attribute: int) -> int:
        for family, values in enumerate(self.families):
            if attribute in values:
                return family
        raise ValueError(f"Unknown attribute: {attribute}")

    def valid_concept(self, concept: tuple[int, ...]) -> bool:
        if not 1 <= len(concept) <= 3 or len(set(concept)) != len(concept):
            return False
        families = [self.attribute_family(value) for value in concept]
        return len(set(families)) == len(families)

    def code_to_concept(self, code: int) -> tuple[int, ...]:
        return tuple(attribute for attribute in range(self.n_attributes) if code & (1 << attribute))

    def all_concepts(self, max_size: int = 3) -> tuple[tuple[int, ...], ...]:
        concepts = []
        for size in range(1, max_size + 1):
            concepts.extend(
                tuple(sorted(values))
                for values in combinations(range(self.n_attributes), size)
                if self.valid_concept(tuple(sorted(values)))
            )
        return tuple(concepts)

    def concepts_for_split(self, split: str) -> tuple[tuple[int, ...], ...]:
        return self.all_concepts(max_size=3)


class TaskGenerator:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.library = ConceptLibrary(config.seed)

    def _positive(self, rng: np.random.Generator, concept: tuple[int, ...]) -> Example:
        attributes = tuple(
            int(rng.choice(values)) for values in self.library.families
        )
        for required in concept:
            family = self.library.attribute_family(required)
            mutable = list(attributes)
            mutable[family] = required
            attributes = tuple(mutable)
        return Example(attributes, True)

    def _negative(
        self,
        rng: np.random.Generator,
        concept: tuple[int, ...],
    ) -> Example:
        if rng.random() < self.config.near_miss_probability:
            positive = self._positive(rng, concept)
            required = int(rng.choice(concept))
            family = self.library.attribute_family(required)
            alternatives = [value for value in self.library.families[family] if value != required]
            attributes = list(positive.attributes)
            attributes[family] = int(rng.choice(alternatives))
            return Example(tuple(attributes), False)
        while True:
            attributes = tuple(
                int(rng.choice(values)) for values in self.library.families
            )
            if not concept_label(concept, attributes):
                return Example(attributes, False)

    def _example(
        self,
        rng: np.random.Generator,
        concept: tuple[int, ...],
        positive: bool | None = None,
    ) -> Example:
        if positive is None:
            positive = bool(rng.integers(0, 2))
        return self._positive(rng, concept) if positive else self._negative(rng, concept)

    def sample(self, split: Split, index: int, seed: int) -> Episode:
        rng = np.random.default_rng(seed)
        concepts = self.library.by_split[split]
        concept = concepts[int(rng.integers(len(concepts)))]
        support = [self._example(rng, concept, index % 2 == 0)]
        support.append(self._example(rng, concept, index % 2 == 1))
        while len(support) < self.config.n_support:
            support.append(self._example(rng, concept))
        if self.config.support_label_noise:
            support = [
                Example(
                    example.attributes,
                    not example.label if rng.random() < self.config.support_label_noise else example.label,
                )
                for example in support
            ]
        n_queries = self.config.n_queries
        labels = [bool(i % 2) for i in range(n_queries)]
        rng.shuffle(labels)
        queries = [self._example(rng, concept, label) for label in labels]
        return Episode(concept, tuple(support), tuple(queries), split, index)
