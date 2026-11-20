from __future__ import annotations

import numpy as np

from .config import DataConfig
from .dcoder_text import TextConceptEncoder
from .tasks import ConceptLibrary, Episode, Example, attribute_family, concept_label
from .tasks import FAMILY_VALUES


RELATION_VALUES = (0, 1, 4, 5, 8, 9, 10, 11)
RELATION_WORDS = ("cs", "cd", "hs", "hd", "nl", "lr", "ts", "td")


def relation_attributes(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        0 if left[0] == right[0] else 1,
        4 if left[1] == right[1] else 5,
        9 if left[2] > right[2] else 8,
        10 if left[3] == right[3] else 11,
    )


class RelationalConceptLibrary(ConceptLibrary):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        allowed = set(RELATION_VALUES)
        self.concepts = tuple(
            concept
            for concept in super().all_concepts(max_size=3)
            if set(concept).issubset(allowed)
        )
        atoms = tuple(concept for concept in self.concepts if len(concept) == 1)
        pairs = [concept for concept in self.concepts if len(concept) == 2]
        triples = [concept for concept in self.concepts if len(concept) == 3]
        rng = np.random.default_rng(seed + 99)
        rng.shuffle(pairs)
        rng.shuffle(triples)
        n_train = int(round(len(pairs) * 0.7))
        n_val = int(round(len(pairs) * 0.15))
        train = atoms + tuple(pairs[:n_train])
        self.by_split = {
            "train": train,
            "seen": train,
            "val": tuple(pairs[n_train:n_train + n_val]),
            "unseen": tuple(pairs[n_train + n_val:]),
            "val_compositional": tuple(triples[:5]),
            "compositional": tuple(triples[5:]),
        }

    def all_concepts(self, max_size: int = 3) -> tuple[tuple[int, ...], ...]:
        return tuple(concept for concept in self.concepts if len(concept) <= max_size)


class RelationalTaskGenerator:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.library = RelationalConceptLibrary(config.seed)

    def _attributes(
        self,
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return relation_attributes(left, right)

    def _example(self, rng: np.random.Generator, concept: tuple[int, ...], positive: bool | None) -> Example:
        if positive is None:
            positive = bool(rng.integers(0, 2))
        if not positive and rng.random() < self.config.near_miss_probability:
            base = self._example(rng, concept, True)
            required = int(rng.choice(concept))
            family = attribute_family(required)
            while True:
                left = list(base.left)
                right = list(base.right)
                if family == 2:
                    sizes = (9, 8) if required == 8 else ((8, 8), (8, 9), (9, 9))[int(rng.integers(3))]
                    left[2], right[2] = sizes
                else:
                    target = left if rng.random() < 0.5 else right
                    target[family] = int(rng.choice(FAMILY_VALUES[family]))
                attributes = self._attributes(tuple(left), tuple(right))
                if attributes[family] != required:
                    return Example(attributes, False, tuple(left), tuple(right))
        while True:
            left = tuple(int(rng.choice(values)) for values in ((0,1,2,3),(4,5,6,7),(8,9),(10,11,12)))
            right = tuple(int(rng.choice(values)) for values in ((0,1,2,3),(4,5,6,7),(8,9),(10,11,12)))
            attributes = self._attributes(left, right)
            if concept_label(concept, attributes) == positive:
                return Example(attributes, positive, left, right)

    def sample(self, split: str, index: int, seed: int) -> Episode:
        rng = np.random.default_rng(seed)
        concepts = self.library.by_split[split]
        concept = concepts[int(rng.integers(len(concepts)))]
        support = [self._example(rng, concept, index % 2 == 0)]
        support.append(self._example(rng, concept, index % 2 == 1))
        while len(support) < self.config.n_support:
            support.append(self._example(rng, concept, None))
        if self.config.support_label_noise:
            support = [
                Example(
                    example.attributes,
                    not example.label if rng.random() < self.config.support_label_noise else example.label,
                    example.left,
                    example.right,
                    example.third,
                )
                for example in support
            ]
        labels = [bool(i % 2) for i in range(self.config.n_queries)]
        rng.shuffle(labels)
        queries = [self._example(rng, concept, label) for label in labels]
        return Episode(concept, tuple(support), tuple(queries), split, index)


class RelationalTextEncoder(TextConceptEncoder):
    def __init__(self, tokenizer, config: DataConfig, mask_token_id: int) -> None:
        super().__init__(tokenizer, config, mask_token_id)
        tokens = [self.none_token] * 13
        for attribute, word in zip(RELATION_VALUES, RELATION_WORDS):
            tokens[attribute] = self._single_token(word)
        self.concept_attribute_tokens = tuple(tokens)
        self.attribute_by_token = {
            self.concept_attribute_tokens[attribute]: attribute
            for attribute in RELATION_VALUES
        }

    def _example_text(self, example: Example) -> str:
        return f"left={self._object(example.left)} right={self._object(example.right)}"
