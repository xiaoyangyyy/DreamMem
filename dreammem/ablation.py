from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, MemoryConfig
from .evaluation import evaluate_trajectories, summarize_results
from .experiment import generate_episodes
from .memory import MEMORY_METHODS
from .tasks import TaskGenerator
from .train import load_checkpoint, resolve_device
from .trajectory import collect_episode_trajectories


def trajectory_variants(config: ExperimentConfig) -> list[tuple[str, str, object, ExperimentConfig]]:
    variants = [("base", "base", "base", config)]
    for steps in (8, 16, 32):
        if steps != config.trajectory.steps:
            variants.append(
                (
                    f"steps_{steps}",
                    "steps",
                    steps,
                    replace(config, trajectory=replace(config.trajectory, steps=steps)),
                )
            )
    for samples in (2, 6):
        if samples != config.trajectory.samples:
            variants.append(
                (
                    f"samples_{samples}",
                    "samples",
                    samples,
                    replace(config, trajectory=replace(config.trajectory, samples=samples)),
                )
            )
    for probability in (0.0, 0.35, 0.7):
        if probability != config.trajectory.revision_probability:
            variants.append(
                (
                    f"revision_{probability}",
                    "revision_probability",
                    probability,
                    replace(
                        config,
                        trajectory=replace(config.trajectory, revision_probability=probability),
                    ),
                )
            )
    return variants


def run_ablations(
    config: ExperimentConfig,
    checkpoint: str | Path,
    evaluation_dir: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint)
    evaluation = Path(evaluation_dir) if evaluation_dir else checkpoint.parent / "evaluation"
    configs_path = evaluation / "best_memory_configs.json"
    raw = json.loads(configs_path.read_text(encoding="utf-8"))
    memory_configs = {method: MemoryConfig(**values) for method, values in raw.items()}
    device = resolve_device(config.device)
    model, _, _ = load_checkpoint(checkpoint, device)
    episodes = generate_episodes(config, "unseen")
    generator = TaskGenerator(config.data)
    output = evaluation / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    results_frames = []
    summary_frames = []
    for name, parameter, value, variant in trajectory_variants(config):
        trajectories = collect_episode_trajectories(
            model,
            variant,
            episodes,
            device,
            seed=config.eval.seed + 501,
        )
        results = evaluate_trajectories(
            trajectories,
            memory_configs,
            generator.library,
            include_symbolic=False,
        )
        results["variant"] = name
        results["parameter"] = parameter
        results["value"] = str(value)
        results_frames.append(results)
        summary = summarize_results(results)
        summary["variant"] = name
        summary["parameter"] = parameter
        summary["value"] = str(value)
        summary["forward_passes_per_episode"] = (
            variant.trajectory.samples
            * variant.trajectory.steps
            * len(variant.trajectory.support_stages)
        )
        summary_frames.append(summary)
        pd.concat(results_frames, ignore_index=True).to_csv(output / "per_episode.csv", index=False)
        pd.concat(summary_frames, ignore_index=True).to_csv(output / "summary.csv", index=False)
        print(json.dumps({"completed_variant": name, "seed_checkpoint": str(checkpoint)}), flush=True)
    return output / "summary.csv"
