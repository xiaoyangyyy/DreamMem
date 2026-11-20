from __future__ import annotations

from enum import Enum

import numpy as np

from .encoding import ATTRIBUTE_BASE, MASK, NIL
from .tasks import ConceptLibrary, N_ATTRIBUTES, code_to_concept, concept_to_code, valid_concept


INVALID_CONCEPT = -1


class RevisionType(str, Enum):
    RETAIN = "retain"
    GENERALIZE = "generalize"
    SPECIALIZE = "specialize"
    SUBSTITUTE = "substitute"
    REPLACE = "replace"
    FORM = "form"
    ERASE = "erase"
    INVALID = "invalid"


def decode_concept_slots(tokens: np.ndarray) -> tuple[int, ...] | None:
    attributes: list[int] = []
    nil_seen = False
    for raw in tokens:
        token = int(raw)
        if token == NIL:
            nil_seen = True
            continue
        if token == MASK:
            return None
        if token < ATTRIBUTE_BASE or token >= ATTRIBUTE_BASE + N_ATTRIBUTES or nil_seen:
            return None
        attributes.append(token - ATTRIBUTE_BASE)
    concept = tuple(sorted(attributes))
    return concept if valid_concept(concept) else None


def encode_concept_slots(tokens: np.ndarray) -> np.ndarray:
    flat = tokens.reshape(-1, tokens.shape[-1])
    codes = np.empty(flat.shape[0], dtype=np.int32)
    for index, row in enumerate(flat):
        concept = decode_concept_slots(row)
        codes[index] = INVALID_CONCEPT if concept is None else concept_to_code(concept)
    return codes.reshape(tokens.shape[:-1])


def revision_type(
    before: int,
    after: int,
    library: ConceptLibrary | None = None,
) -> RevisionType:
    library = library or ConceptLibrary()
    before_valid = before != INVALID_CONCEPT
    after_valid = after != INVALID_CONCEPT
    if not before_valid and not after_valid:
        return RevisionType.INVALID
    if not before_valid:
        return RevisionType.FORM
    if not after_valid:
        return RevisionType.ERASE
    if before == after:
        return RevisionType.RETAIN
    previous = set(library.code_to_concept(before))
    current = set(library.code_to_concept(after))
    if current < previous:
        return RevisionType.GENERALIZE
    if current > previous:
        return RevisionType.SPECIALIZE
    if not current.intersection(previous):
        return RevisionType.REPLACE
    return RevisionType.SUBSTITUTE


def transition_matrix(codes: np.ndarray) -> np.ndarray:
    if codes.shape[-1] < 2:
        return np.empty(codes.shape[:-1] + (0,), dtype=np.int8)
    types = list(RevisionType)
    indices = {kind: index for index, kind in enumerate(types)}
    output = np.empty(codes.shape[:-1] + (codes.shape[-1] - 1,), dtype=np.int8)
    for index in np.ndindex(codes.shape[:-1]):
        before = codes[index]
        for step in range(codes.shape[-1] - 1):
            output[index + (step,)] = indices[revision_type(int(before[step]), int(before[step + 1]))]
    return output


def revision_summary(codes: np.ndarray) -> dict[str, float]:
    codes = np.asarray(codes)
    valid = codes != INVALID_CONCEPT
    changes = np.zeros_like(valid)
    if codes.shape[-1] > 1:
        changes[..., 1:] = codes[..., 1:] != codes[..., :-1]
    sizes = np.zeros_like(codes, dtype=np.float32)
    for code in np.unique(codes[valid]):
        sizes[codes == code] = len(code_to_concept(int(code)))
    transitions = transition_matrix(codes)
    summary = {
        "valid_fraction": float(valid.mean()),
        "change_rate": float(changes.mean()),
        "late_change_rate": float(changes[..., changes.shape[-1] // 2 :].mean()),
        "mean_concept_size": float(sizes[valid].mean()) if valid.any() else 0.0,
    }
    for index, kind in enumerate(RevisionType):
        summary[f"{kind.value}_rate"] = float((transitions == index).mean()) if transitions.size else 0.0
    return summary
