from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, save_config
from .dcoder import DCoderAdapter, load_dcoder
from .dcoder_text import TextConceptEncoder
from .evaluation import (
    bootstrap_intervals,
    concept_cluster_intervals,
    evaluate_trajectories,
    paired_comparisons,
    summarize_results,
    tune_memory_configs,
)
from .experiment import (
    generate_episodes,
    load_trajectories,
    save_trajectories,
)
from .plots import plot_risk_coverage, plot_summary, plot_training
from .probe import run_probe
from .task_factory import make_task_components
from .tasks import TaskGenerator
from .trajectory import EpisodeTrajectory, collect_episode_trajectories


def evaluate_dcoder(
    config: ExperimentConfig,
    model_path: str | Path,
    output_dir: str | Path,
    reuse_trajectories: bool = False,
) -> dict[str, Path]:
    model_path = Path(model_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    device = torch.device("cuda")
    tokenizer, model = load_dcoder(model_path, device)
    adapter = DCoderAdapter(model)
    generator, encoder = make_task_components(
        config,
        tokenizer,
        int(model.generation_config.mask_token_id),
    )
    trajectories_by_split: dict[str, list[EpisodeTrajectory]] = {}
    timings = {}
    for split in ["val", "val_compositional", *config.eval.splits]:
        path = output / "trajectories" / f"{split}.pt"
        if reuse_trajectories and path.exists():
            trajectories = load_trajectories(path)
        else:
            episodes = generate_episodes(config, split)
            started = time.time()
            trajectories = collect_episode_trajectories(
                adapter,
                config,
                episodes,
                device,
                seed=config.eval.seed + sum(ord(char) for char in split),
                encoder=encoder,
                mask_token_id=encoder.mask_token_id,
                concept_codes_fn=encoder.concept_codes,
            )
            timings[split] = time.time() - started
            save_trajectories(path, trajectories)
        trajectories_by_split[split] = trajectories
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tuning, tuning_history = tune_memory_configs(
        trajectories_by_split["val"],
        config.memory,
        generator.library,
        config.eval.tuning_trials,
        config.eval.seed + 81,
        trajectories_by_split["val_compositional"],
    )
    tuning_history.to_csv(output / "tuning_history.csv", index=False)
    (output / "best_memory_configs.json").write_text(
        json.dumps({method: asdict(value) for method, value in tuning.items()}, indent=2),
        encoding="utf-8",
    )
    all_results = []
    metadata_path = model_path / "experiment_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    for split in config.eval.splits:
        results = evaluate_trajectories(
            trajectories_by_split[split],
            tuning,
            generator.library,
        )
        results["seed"] = metadata.get("seed", config.train.seed)
        results.to_csv(output / f"results_{split}.csv", index=False)
        all_results.append(results)
    combined = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(combined)
    intervals = bootstrap_intervals(combined, config.eval.bootstrap_samples, config.eval.seed + 82)
    comparisons = paired_comparisons(
        combined,
        samples=config.eval.bootstrap_samples,
        seed=config.eval.seed + 83,
    )
    clusters = concept_cluster_intervals(
        combined,
        config.eval.bootstrap_samples,
        config.eval.seed + 84,
    )
    summary.to_csv(output / "summary.csv", index=False)
    intervals.to_csv(output / "intervals.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    clusters.to_csv(output / "concept_cluster_intervals.csv", index=False)
    probe = run_probe(trajectories_by_split["val"], trajectories_by_split["unseen"])
    probe.to_csv(output / "semantic_revision_probe.csv", index=False)
    plot_summary(summary, output / "metrics.png")
    plot_risk_coverage(combined, output / "risk_coverage.png")
    history_path = model_path.parent / "history.csv"
    if history_path.exists():
        plot_training(history_path, output / "training.png")
    runtime = {
        "model_path": str(model_path),
        "checkpoint_step": metadata.get("step"),
        "trajectory_seconds": timings,
        "forward_passes_per_episode": (
            config.trajectory.samples
            * config.trajectory.steps
            * len(config.trajectory.support_stages)
        ),
        "episodes_per_split": config.eval.episodes,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return {
        "summary": output / "summary.csv",
        "intervals": output / "intervals.csv",
        "comparisons": output / "paired_comparisons.csv",
        "probe": output / "semantic_revision_probe.csv",
    }
