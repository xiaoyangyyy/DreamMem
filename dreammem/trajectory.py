from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .encoding import MASK, EncodedBatch, SequenceEncoder
from .model import MaskedDiffusionTransformer
from .semantics import encode_concept_slots
from .tasks import Episode


@dataclass
class StageTrajectory:
    support_count: int
    concept_tokens: np.ndarray
    concept_codes: np.ndarray
    concept_confidence: np.ndarray
    query_tokens: np.ndarray
    query_confidence: np.ndarray


@dataclass
class EpisodeTrajectory:
    episode: Episode
    stages: list[StageTrajectory]


@dataclass
class DenoisingRecord:
    concept_tokens: torch.Tensor
    concept_confidence: torch.Tensor
    query_tokens: torch.Tensor
    query_confidence: torch.Tensor


@torch.no_grad()
def denoise_with_revisions(
    model: MaskedDiffusionTransformer,
    batch: EncodedBatch,
    encoder: SequenceEncoder,
    steps: int,
    temperature: float,
    revision_probability: float,
    generator: torch.Generator,
    mask_token_id: int = MASK,
) -> DenoisingRecord:
    device = batch.tokens.device
    free_mask = batch.concept_mask | batch.query_mask
    state = batch.tokens.masked_fill(free_mask, mask_token_id)
    batch_size = state.shape[0]
    concept_tokens = torch.empty(
        (batch_size, steps, len(encoder.concept_positions)),
        dtype=torch.int32,
        device=device,
    )
    query_tokens = torch.empty(
        (batch_size, steps, len(encoder.query_label_positions)),
        dtype=torch.int32,
        device=device,
    )
    concept_confidence = torch.empty(
        concept_tokens.shape,
        dtype=torch.float16,
        device=device,
    )
    query_confidence = torch.empty(
        query_tokens.shape,
        dtype=torch.float16,
        device=device,
    )
    committed_confidence = torch.zeros_like(state, dtype=torch.float32)
    n_free = int(free_mask[0].sum())
    for step in range(steps):
        ratio = 1.0 - step / steps
        ratios = torch.full((batch_size,), ratio, device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(state, ratios)
        free_positions = free_mask[0]
        full_shape = logits.shape
        scaled_logits = logits[:, free_positions].float() / max(temperature, 1e-4)
        del logits
        probabilities = scaled_logits.softmax(dim=-1)
        gumbel = torch.rand(
            full_shape,
            device=device,
            generator=generator,
        ).clamp_min_(1e-6)
        gumbel = gumbel[:, free_positions]
        gumbel.log_().neg_().clamp_min_(1e-6).log_().neg_()
        predicted_free = scaled_logits.add_(gumbel).argmax(dim=-1)
        del gumbel
        confidence_free = probabilities.gather(-1, predicted_free.unsqueeze(-1)).squeeze(-1)
        del scaled_logits, probabilities
        predicted = batch.tokens.clone()
        predicted[:, free_positions] = predicted_free
        confidence = torch.zeros_like(state, dtype=torch.float32)
        confidence[:, free_positions] = confidence_free
        step_concepts = predicted[:, encoder.concept_positions]
        step_queries = predicted[:, encoder.query_label_positions]
        concept_tokens[:, step] = step_concepts.to(torch.int32)
        query_tokens[:, step] = step_queries.to(torch.int32)
        concept_confidence[:, step] = confidence[:, encoder.concept_positions].to(torch.float16)
        query_confidence[:, step] = confidence[:, encoder.query_label_positions].to(torch.float16)
        if step == steps - 1:
            state = torch.where(free_mask, predicted, batch.tokens)
            continue
        progress = (step + 1) / steps
        desired = max(1, int(np.ceil(progress * n_free)))
        currently_committed = ((state != mask_token_id) & free_mask).sum(dim=1)
        update = (state == mask_token_id) & free_mask
        for row in range(batch_size):
            missing = int(desired - currently_committed[row])
            if missing > 0:
                masked_positions = torch.nonzero(update[row], as_tuple=False).flatten()
                order = torch.argsort(confidence[row, masked_positions], descending=True)
                update[row, masked_positions[order[missing:]]] = False
            committed = ((state[row] != mask_token_id) & free_mask[row]).nonzero(as_tuple=False).flatten()
            if len(committed) and revision_probability > 0:
                revise_count = int(len(committed) * revision_probability * (1.0 - progress))
                if revise_count:
                    jitter = torch.rand(
                        len(committed),
                        device=device,
                        generator=generator,
                    ) * 0.05
                    order = torch.argsort(committed_confidence[row, committed] + jitter)
                    update[row, committed[order[:revise_count]]] = True
        state = torch.where(update, predicted, state)
        committed_confidence = torch.where(update, confidence, committed_confidence)
    return DenoisingRecord(
        concept_tokens,
        concept_confidence,
        query_tokens,
        query_confidence,
    )


def collect_episode_trajectories(
    model: MaskedDiffusionTransformer,
    config: ExperimentConfig,
    episodes: list[Episode],
    device: torch.device,
    seed: int,
    encoder=None,
    mask_token_id: int = MASK,
    concept_codes_fn=encode_concept_slots,
) -> list[EpisodeTrajectory]:
    model.eval()
    if encoder is None:
        encoder = SequenceEncoder(config.data)
    trajectory = config.trajectory
    samples = trajectory.samples
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    output = [EpisodeTrajectory(episode, []) for episode in episodes]
    for support_count in trajectory.support_stages:
        expanded = [episode for episode in episodes for _ in range(samples)]
        print(
            f"Trajectory seed={seed} support={support_count}: "
            f"0/{len(expanded)}",
            flush=True,
        )
        concept_tokens = np.empty(
            (len(expanded), trajectory.steps, 3), dtype=np.int32
        )
        concept_confidence = np.empty_like(concept_tokens, dtype=np.float16)
        query_tokens = np.empty(
            (len(expanded), trajectory.steps, config.data.n_queries), dtype=np.int32
        )
        query_confidence = np.empty_like(query_tokens, dtype=np.float16)
        for start in range(0, len(expanded), config.eval.batch_size):
            stop = min(start + config.eval.batch_size, len(expanded))
            subset = expanded[start:stop]
            batch = encoder.encode_batch(subset, support_count, device=device)
            record = denoise_with_revisions(
                model,
                batch,
                encoder,
                trajectory.steps,
                trajectory.temperature,
                trajectory.revision_probability,
                generator,
                mask_token_id,
            )
            concept_tokens[start:stop] = record.concept_tokens.cpu().numpy()
            concept_confidence[start:stop] = record.concept_confidence.cpu().numpy()
            query_tokens[start:stop] = record.query_tokens.cpu().numpy()
            query_confidence[start:stop] = record.query_confidence.cpu().numpy()
            if stop % 32 == 0 or stop == len(expanded):
                print(
                    f"Trajectory seed={seed} support={support_count}: "
                    f"{stop}/{len(expanded)}",
                    flush=True,
                )
        codes = concept_codes_fn(concept_tokens)
        for episode_index in range(len(episodes)):
            begin = episode_index * samples
            end = begin + samples
            output[episode_index].stages.append(
                StageTrajectory(
                    support_count=support_count,
                    concept_tokens=concept_tokens[begin:end],
                    concept_codes=codes[begin:end],
                    concept_confidence=concept_confidence[begin:end],
                    query_tokens=query_tokens[begin:end],
                    query_confidence=query_confidence[begin:end],
                )
            )
    return output
