from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DataConfig
from .tasks import Episode, N_ATTRIBUTES


PAD = 0
MASK = 1
BOS = 2
SEP = 3
NIL = 4
NO = 5
YES = 6
ATTRIBUTE_BASE = 7
VOCAB_SIZE = ATTRIBUTE_BASE + N_ATTRIBUTES
EXAMPLE_WIDTH = 6
CONCEPT_POSITIONS = (1, 2, 3)


@dataclass
class EncodedBatch:
    tokens: torch.Tensor
    variable_mask: torch.Tensor
    concept_mask: torch.Tensor
    query_mask: torch.Tensor
    support_mask: torch.Tensor
    active_support: torch.Tensor
    episodes: list[Episode]


class SequenceEncoder:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.n_examples = config.n_support + config.n_queries
        self.length = 4 + self.n_examples * EXAMPLE_WIDTH
        self.support_starts = [4 + i * EXAMPLE_WIDTH for i in range(config.n_support)]
        self.query_starts = [
            4 + (config.n_support + i) * EXAMPLE_WIDTH
            for i in range(config.n_queries)
        ]
        self.support_label_positions = [start + 4 for start in self.support_starts]
        self.query_label_positions = [start + 4 for start in self.query_starts]
        self.concept_positions = list(CONCEPT_POSITIONS)

    def encode(self, episode: Episode, support_count: int | None = None) -> np.ndarray:
        support_count = self.config.n_support if support_count is None else support_count
        if not 1 <= support_count <= self.config.n_support:
            raise ValueError(f"Invalid support count: {support_count}")
        tokens = np.full(self.length, PAD, dtype=np.int64)
        tokens[0] = BOS
        concept = list(episode.concept) + [None] * (3 - len(episode.concept))
        for position, attribute in zip(self.concept_positions, concept):
            tokens[position] = NIL if attribute is None else ATTRIBUTE_BASE + attribute
        for index, start in enumerate(self.support_starts):
            if index >= support_count:
                continue
            example = episode.support[index]
            for family, attribute in enumerate(example.attributes):
                tokens[start + family] = ATTRIBUTE_BASE + attribute
            tokens[start + 4] = YES if example.label else NO
            tokens[start + 5] = SEP
        for index, start in enumerate(self.query_starts):
            example = episode.queries[index]
            for family, attribute in enumerate(example.attributes):
                tokens[start + family] = ATTRIBUTE_BASE + attribute
            tokens[start + 4] = YES if example.label else NO
            tokens[start + 5] = SEP
        return tokens

    def encode_batch(
        self,
        episodes: list[Episode],
        support_counts: list[int] | int | None = None,
        device: torch.device | str = "cpu",
    ) -> EncodedBatch:
        if support_counts is None:
            support_counts = [self.config.n_support] * len(episodes)
        elif isinstance(support_counts, int):
            support_counts = [support_counts] * len(episodes)
        tokens = np.stack(
            [self.encode(episode, count) for episode, count in zip(episodes, support_counts)]
        )
        variable = np.zeros_like(tokens, dtype=bool)
        concept = np.zeros_like(variable)
        query = np.zeros_like(variable)
        support = np.zeros_like(variable)
        active = np.zeros_like(variable)
        variable[:, self.concept_positions] = True
        concept[:, self.concept_positions] = True
        query[:, self.query_label_positions] = True
        variable[:, self.query_label_positions] = True
        for row, count in enumerate(support_counts):
            positions = self.support_label_positions[:count]
            support[row, positions] = True
            variable[row, positions] = True
            for start in self.support_starts[:count]:
                active[row, start : start + EXAMPLE_WIDTH] = True
        return EncodedBatch(
            tokens=torch.as_tensor(tokens, device=device),
            variable_mask=torch.as_tensor(variable, device=device),
            concept_mask=torch.as_tensor(concept, device=device),
            query_mask=torch.as_tensor(query, device=device),
            support_mask=torch.as_tensor(support, device=device),
            active_support=torch.as_tensor(active, device=device),
            episodes=episodes,
        )
