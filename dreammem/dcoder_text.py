from __future__ import annotations

import numpy as np
import torch

from .config import DataConfig
from .encoding import EncodedBatch
from .semantics import INVALID_CONCEPT
from .tasks import Episode, concept_to_code, valid_concept


ATTRIBUTE_WORDS = (
    "red", "green", "blue", "yellow",
    "circle", "square", "triangle", "star",
    "small", "large",
    "plain", "lines", "dots",
)


class TextConceptEncoder:
    def __init__(self, tokenizer, config: DataConfig, mask_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.mask_token_id = mask_token_id
        self.spaced_attributes = {
            word for word in ATTRIBUTE_WORDS
            if len(self.tokenizer(word, add_special_tokens=False).input_ids) != 1
        }
        self.attribute_tokens = tuple(self._single_token(word) for word in ATTRIBUTE_WORDS)
        self.concept_attribute_tokens = self.attribute_tokens
        self.attribute_by_token = {
            token: attribute for attribute, token in enumerate(self.attribute_tokens)
        }
        self.yes_token = self._single_token("yes")
        self.no_token = self._single_token("no")
        self.none_token = self._single_token("none")
        self.concept_positions: list[int] = []
        self.query_label_positions: list[int] = []
        self.support_label_positions: list[int] = []

    def _single_token(self, text: str) -> int:
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(tokens) != 1:
            tokens = self.tokenizer(f" {text}", add_special_tokens=False).input_ids
        if len(tokens) != 1:
            raise ValueError(f"Expected one token for {text!r}, got {tokens}")
        return int(tokens[0])

    def _text(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _attribute_value(self, index: int) -> str:
        word = ATTRIBUTE_WORDS[index]
        return f" {word}" if word in self.spaced_attributes else word

    def _object(self, attributes: tuple[int, ...]) -> str:
        return (
            f"color={self._attribute_value(attributes[0])} "
            f"shape={self._attribute_value(attributes[1])} "
            f"size={self._attribute_value(attributes[2])} "
            f"texture={self._attribute_value(attributes[3])}"
        )

    def _example_text(self, example) -> str:
        return self._object(example.attributes)

    def encode_episode(self, episode: Episode, support_count: int) -> np.ndarray:
        tokens: list[int] = []
        concept_positions: list[int] = []
        query_positions: list[int] = []
        support_positions: list[int] = []
        tokens.extend(self._text("Form a reusable concept from the experiences.\n"))
        for index, example in enumerate(episode.support[:support_count]):
            tokens.extend(self._text(f"Experience {index + 1}: {self._example_text(example)} label="))
            support_positions.append(len(tokens))
            tokens.append(self.yes_token if example.label else self.no_token)
            tokens.extend(self._text("\n"))
        tokens.extend(self._text("Concept candidates:"))
        for slot in range(3):
            tokens.extend(self._text(f" c{slot + 1}="))
            concept_positions.append(len(tokens))
            if slot < len(episode.concept):
                tokens.append(self.concept_attribute_tokens[episode.concept[slot]])
            else:
                tokens.append(self.none_token)
        tokens.extend(self._text("\n"))
        for index, query in enumerate(episode.queries):
            tokens.extend(self._text(f"Query {index + 1}: {self._example_text(query)} label="))
            query_positions.append(len(tokens))
            tokens.append(self.yes_token if query.label else self.no_token)
            tokens.extend(self._text("\n"))
        self.concept_positions = concept_positions
        self.query_label_positions = query_positions
        self.support_label_positions = support_positions
        return np.asarray(tokens, dtype=np.int64)

    def encode_batch(
        self,
        episodes: list[Episode],
        support_counts: int | list[int] | None = None,
        device: torch.device | str = "cpu",
    ) -> EncodedBatch:
        if support_counts is None:
            support_counts = self.config.n_support
        if isinstance(support_counts, int):
            support_counts = [support_counts] * len(episodes)
        encoded = [
            self.encode_episode(episode, count)
            for episode, count in zip(episodes, support_counts)
        ]
        maximum = max(len(sequence) for sequence in encoded)
        tokens = np.full((len(encoded), maximum), self.mask_token_id, dtype=np.int64)
        variable = np.zeros_like(tokens, dtype=bool)
        concept = np.zeros_like(variable)
        query = np.zeros_like(variable)
        support = np.zeros_like(variable)
        active = np.zeros_like(variable)
        for row, sequence in enumerate(encoded):
            tokens[row, : len(sequence)] = sequence
            active[row, : len(sequence)] = True
        for row, sequence in enumerate(encoded):
            self.encode_episode(episodes[row], support_counts[row])
            concept[row, self.concept_positions] = True
            query[row, self.query_label_positions] = True
            support[row, self.support_label_positions] = True
        variable = concept | query | support
        return EncodedBatch(
            tokens=torch.as_tensor(tokens, device=device),
            variable_mask=torch.as_tensor(variable, device=device),
            concept_mask=torch.as_tensor(concept, device=device),
            query_mask=torch.as_tensor(query, device=device),
            support_mask=torch.as_tensor(support, device=device),
            active_support=torch.as_tensor(active, device=device),
            episodes=episodes,
        )

    def decode_concept(self, tokens: np.ndarray) -> tuple[int, ...] | None:
        attributes = []
        none_seen = False
        for raw in tokens:
            token = int(raw)
            if token == self.none_token:
                none_seen = True
            elif token in self.attribute_by_token and not none_seen:
                attributes.append(self.attribute_by_token[token])
            else:
                return None
        concept = tuple(sorted(attributes))
        return concept if valid_concept(concept) else None

    def concept_codes(self, tokens: np.ndarray) -> np.ndarray:
        flat = tokens.reshape(-1, tokens.shape[-1])
        codes = np.empty(len(flat), dtype=np.int64)
        for index, row in enumerate(flat):
            concept = self.decode_concept(row)
            codes[index] = INVALID_CONCEPT if concept is None else concept_to_code(concept)
        return codes.reshape(tokens.shape[:-1])
