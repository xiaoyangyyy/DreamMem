from __future__ import annotations

import numpy as np
import torch

from .semantics import INVALID_CONCEPT
from .trajectory import EpisodeTrajectory, StageTrajectory


@torch.no_grad()
def collect_ar_trajectories(
    model,
    config,
    episodes,
    device,
    encoder,
    seed: int,
) -> list[EpisodeTrajectory]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    output = [EpisodeTrajectory(episode, []) for episode in episodes]
    samples = 14
    for support_count in config.trajectory.support_stages:
        concept_tokens = np.empty((len(episodes) * samples, 1, 3), dtype=np.int32)
        concept_confidence = np.empty_like(concept_tokens, dtype=np.float16)
        query_tokens = np.empty((len(episodes) * samples, 1, config.data.n_queries), dtype=np.int32)
        query_confidence = np.empty_like(query_tokens, dtype=np.float16)
        for episode_start in range(0, len(episodes), config.eval.batch_size):
            subset = episodes[episode_start:episode_start + config.eval.batch_size]
            for sample_start in range(0, samples, 2):
                current_samples = min(2, samples - sample_start)
                expanded = [episode for episode in subset for _ in range(current_samples)]
                batch = encoder.encode_batch(expanded, support_count, device=device)
                free_mask = batch.concept_mask | batch.query_mask
                state = batch.tokens.masked_fill(free_mask, encoder.mask_token_id)
                positions = sorted(encoder.concept_positions + encoder.query_label_positions)
                position_index = {position: index for index, position in enumerate(positions)}
                concept_values = torch.empty((len(expanded), 3), dtype=torch.int32, device=device)
                concept_probabilities = torch.empty((len(expanded), 3), dtype=torch.float16, device=device)
                query_values = torch.empty((len(expanded), config.data.n_queries), dtype=torch.int32, device=device)
                query_probabilities = torch.empty((len(expanded), config.data.n_queries), dtype=torch.float16, device=device)
                for position in positions:
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(state, torch.ones(len(expanded), device=device))
                        current = logits[:, position].float() / config.trajectory.temperature
                    probabilities = current.softmax(dim=-1)
                    noise = torch.rand(probabilities.shape, device=device, generator=generator).clamp_min_(1e-6)
                    noise.log_().neg_().clamp_min_(1e-6).log_().neg_()
                    predicted = current.add_(noise).argmax(dim=-1)
                    confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                    state[:, position] = predicted
                    if position in encoder.concept_positions:
                        index = encoder.concept_positions.index(position)
                        concept_values[:, index] = predicted.to(torch.int32)
                        concept_probabilities[:, index] = confidence.to(torch.float16)
                    else:
                        index = encoder.query_label_positions.index(position)
                        query_values[:, index] = predicted.to(torch.int32)
                        query_probabilities[:, index] = confidence.to(torch.float16)
                destination_start = episode_start * samples + sample_start * len(subset)
                destination_stop = destination_start + len(expanded)
                concept_tokens[destination_start:destination_stop, 0] = concept_values.cpu().numpy()
                concept_confidence[destination_start:destination_stop, 0] = concept_probabilities.cpu().numpy()
                query_tokens[destination_start:destination_stop, 0] = query_values.cpu().numpy()
                query_confidence[destination_start:destination_stop, 0] = query_probabilities.cpu().numpy()
        codes = encoder.concept_codes(concept_tokens)
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


@torch.no_grad()
def collect_mlm_trajectories(
    model,
    config,
    episodes,
    device,
    encoder,
    seed: int,
) -> list[EpisodeTrajectory]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    output = [EpisodeTrajectory(episode, []) for episode in episodes]
    samples = 6
    for support_count in config.trajectory.support_stages:
        concept_tokens = np.empty((len(episodes) * samples, 1, 3), dtype=np.int32)
        concept_confidence = np.empty_like(concept_tokens, dtype=np.float16)
        query_tokens = np.empty((len(episodes) * samples, 1, config.data.n_queries), dtype=np.int32)
        query_confidence = np.empty_like(query_tokens, dtype=np.float16)
        for start in range(0, len(episodes), config.eval.batch_size):
            subset = episodes[start:start + config.eval.batch_size]
            expanded = [episode for episode in subset for _ in range(samples)]
            batch = encoder.encode_batch(expanded, support_count, device=device)
            free_mask = batch.concept_mask | batch.query_mask
            inputs = batch.tokens.masked_fill(free_mask, encoder.mask_token_id)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(
                    inputs,
                    torch.ones(len(expanded), device=device),
                    positions=encoder.concept_positions + encoder.query_label_positions,
                )
                current = logits.float() / config.trajectory.temperature
            probabilities = current.softmax(dim=-1)
            noise = torch.rand(probabilities.shape, device=device, generator=generator).clamp_min_(1e-6)
            noise.log_().neg_().clamp_min_(1e-6).log_().neg_()
            predicted = current.add_(noise).argmax(dim=-1)
            confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            concept_values = predicted[:, :3]
            query_values = predicted[:, 3:]
            concept_tokens[start * samples:(start + len(subset)) * samples, 0] = concept_values.cpu().numpy()
            query_tokens[start * samples:(start + len(subset)) * samples, 0] = query_values.cpu().numpy()
            concept_confidence[start * samples:(start + len(subset)) * samples, 0] = confidence[:, :3].to(torch.float16).cpu().numpy()
            query_confidence[start * samples:(start + len(subset)) * samples, 0] = confidence[:, 3:].to(torch.float16).cpu().numpy()
        codes = encoder.concept_codes(concept_tokens)
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
