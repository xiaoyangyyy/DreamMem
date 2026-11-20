from __future__ import annotations

from dataclasses import replace

from .config import DataConfig
from .dcoder_text import ATTRIBUTE_WORDS, TextConceptEncoder
from .tasks import TaskGenerator


NATURAL_DOMAINS = ("history", "traffic", "organization", "device", "abstract")
NATURAL_VOCABULARIES = (
    ("crimson", "emerald", "azure", "golden", "ring", "square", "wedge", "star", "small", "large", "plain", "lined", "dotted"),
    ("scarlet", "verdant", "cyan", "amber", "circular", "rectangular", "triangular", "radial", "compact", "extensive", "smooth", "striped", "spotted"),
    ("ruby", "jade", "navy", "ochre", "circle", "cube", "pyramid", "asterisk", "minor", "major", "blank", "linear", "speckled"),
    ("maroon", "olive", "cobalt", "saffron", "loop", "block", "angular", "stellar", "limited", "expanded", "flat", "grooved", "dotted"),
    ATTRIBUTE_WORDS,
)
NATURAL_PATTERNS = (
    "The legion carries the {} banner, forms a {}, keeps a {} garrison, and bears a {} seal.",
    "The intersection uses the {} signal, a {} route layout, a {} network, and {} road markings.",
    "The organization assigns the {} division to the {} structure, with a {} team and {} case marks.",
    "The device shows the {} indicator, uses a {} chassis, has {} capacity, and provides a {} interface.",
    "The record has alpha={}, beta={}, gamma={}, and delta={}.",
)


class NaturalTaskGenerator(TaskGenerator):
    def sample(self, split: str, index: int, seed: int):
        episode = super().sample(split, index, seed)
        if split in ("train", "seen"):
            template = index % 2
        elif split in ("val", "val_compositional"):
            template = 2
        elif split == "unseen":
            template = 3
        else:
            template = 4
        return replace(episode, template=template)


class NaturalTextEncoder(TextConceptEncoder):
    def encode_episode(self, episode, support_count: int):
        self._template = episode.template
        return super().encode_episode(episode, support_count)

    def _object(self, attributes: tuple[int, ...]) -> str:
        vocabulary = NATURAL_VOCABULARIES[self._template]
        values = [vocabulary[attribute] for attribute in attributes]
        return NATURAL_PATTERNS[self._template].format(*values)
