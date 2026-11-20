from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .encoding import MASK, EncodedBatch


@dataclass
class DiffusionBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    mask_ratio: torch.Tensor


def scheduled_mask_ratio(
    time: torch.Tensor,
    schedule: str = "cosine",
    minimum: float = 0.05,
) -> torch.Tensor:
    if schedule == "cosine":
        ratio = torch.cos(time * math.pi / 2)
    elif schedule == "linear":
        ratio = 1.0 - time
    elif schedule == "uniform":
        ratio = torch.ones_like(time)
    else:
        raise ValueError(f"Unknown mask schedule: {schedule}")
    return ratio.clamp(min=minimum, max=1.0)


def corrupt_batch(
    batch: EncodedBatch,
    generator: torch.Generator,
    schedule: str = "cosine",
    minimum: float = 0.05,
    mask_token_id: int = MASK,
) -> DiffusionBatch:
    time = torch.rand(batch.tokens.shape[0], device=batch.tokens.device, generator=generator)
    ratios = scheduled_mask_ratio(time, schedule, minimum)
    random = torch.rand(
        batch.tokens.shape,
        device=batch.tokens.device,
        generator=generator,
    )
    loss_mask = (random < ratios[:, None]) & batch.variable_mask
    empty = ~loss_mask.any(dim=1)
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        for row in rows:
            candidates = torch.nonzero(batch.variable_mask[row], as_tuple=False).flatten()
            choice = torch.randint(
                len(candidates),
                (1,),
                device=batch.tokens.device,
                generator=generator,
            )
            loss_mask[row, candidates[choice]] = True
    inputs = batch.tokens.masked_fill(loss_mask, mask_token_id)
    return DiffusionBatch(inputs, batch.tokens, loss_mask, ratios)


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = loss_mask.reshape(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[selected].float(),
        targets.reshape(-1)[selected],
        reduction="none",
    )
    if position_weights is None:
        return losses.mean()
    weights = position_weights.reshape(-1)[selected].float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def masked_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    selection: torch.Tensor,
) -> torch.Tensor:
    count = selection.sum()
    if count.item() == 0:
        return logits.new_tensor(float("nan"))
    correct = (logits.argmax(dim=-1) == targets) & selection
    return correct.sum().float() / count.float()
