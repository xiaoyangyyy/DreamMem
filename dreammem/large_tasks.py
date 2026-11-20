from __future__ import annotations

from .config import DataConfig
from .dcoder_text import TextConceptEncoder
from .tasks import ConceptLibrary, TaskGenerator


LARGE_FAMILIES = tuple(tuple(range(4 * family, 4 * family + 4)) for family in range(8))
LARGE_WORDS = (
    "red", "green", "blue", "yellow",
    "circle", "square", "triangle", "star",
    "small", "large", "plain", "lines", "dots",
    "white", "black", "purple", "orange",
    "rectangle", "diamond", "oval", "hex",
    "tiny", "huge", "smooth", "rough", "strip", "spots",
    "metal", "wood", "fabric", "glass", "cloth",
)


class LargeTaskGenerator(TaskGenerator):
    def __init__(self, config: DataConfig) -> None:
        super().__init__(config)
        self.library = ConceptLibrary(config.seed, families=LARGE_FAMILIES)


class LargeTextEncoder(TextConceptEncoder):
    def __init__(self, tokenizer, config: DataConfig, mask_token_id: int) -> None:
        super().__init__(tokenizer, config, mask_token_id)
        self.library = ConceptLibrary(config.seed, families=LARGE_FAMILIES)
        self.attribute_tokens = tuple(self._single_token(word) for word in LARGE_WORDS)
        self.concept_attribute_tokens = self.attribute_tokens
        self.attribute_by_token = {
            token: attribute for attribute, token in enumerate(self.attribute_tokens)
        }

    def _object(self, attributes: tuple[int, ...]) -> str:
        return " ".join(
            f"f{family}={LARGE_WORDS[attributes[family]]}" for family in range(len(LARGE_FAMILIES))
        )

    def decode_concept(self, tokens):
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
        return concept if self.library.valid_concept(concept) else None
