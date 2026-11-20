from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, MemoryConfig, config_to_dict, save_config
from .evaluation import (
    bootstrap_intervals,
    concept_cluster_intervals,
    evaluate_trajectories,
    paired_comparisons,
    summarize_results,
    tune_memory_configs,
)
from .memory import MEMORY_METHODS
from .plots import plot_risk_coverage, plot_summary, plot_training
from .probe import run_probe
from .tasks import TaskGenerator
from .train import load_checkpoint, resolve_device, train_model
from .trajectory import EpisodeTrajectory, collect_episode_trajectories


def generate_episodes(config: ExperimentConfig, split: str) -> list:
    from .task_factory import make_task_generator

    generator = make_task_generator(config)
    return [
        generator.sample(split, index, config.eval.seed + index)
        for index in range(config.eval.episodes)
    ]


def save_trajectories(path: Path, trajectories: list[EpisodeTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, path)


def load_trajectories(path: Path) -> list[EpisodeTrajectory]:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    reuse_trajectories: bool = False,
) -> dict[str, Path]:
    checkpoint = Path(checkpoint)
    output = Path(output_dir) if output_dir else checkpoint.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    device = resolve_device(config.device)
    model, _, metadata = load_checkpoint(checkpoint, device)
    generator = TaskGenerator(config.data)
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
                model,
                config,
                episodes,
                device,
                seed=config.eval.seed + sum(ord(char) for char in split),
            )
            timings[split] = time.time() - started
            save_trajectories(path, trajectories)
        trajectories_by_split[split] = trajectories
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
    intervals = bootstrap_intervals(
        combined,
        config.eval.bootstrap_samples,
        config.eval.seed + 82,
    )
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
    clusters.to_csv(output / "concept_cluster_intervals.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    probe = run_probe(trajectories_by_split["val"], trajectories_by_split["unseen"])
    probe.to_csv(output / "semantic_revision_probe.csv", index=False)
    plot_summary(summary, output / "metrics.png")
    plot_risk_coverage(combined, output / "risk_coverage.png")
    history_path = checkpoint.parent / "history.csv"
    if history_path.exists():
        plot_training(history_path, output / "training.png")
    runtime = {
        "checkpoint": str(checkpoint),
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


def aggregate_runs(output_dir: str | Path) -> pd.DataFrame:
    root = Path(output_dir)

    def collect(pattern: str) -> tuple[pd.DataFrame, list[Path]]:
        paths = list(root.glob(pattern))
        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            seed = next(part for part in path.parts if part.startswith("seed_"))
            frame["run_seed"] = int(seed.split("_")[-1])
            frames.append(frame)
        return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), paths)

    def aggregate_frame(
        combined: pd.DataFrame,
        groups: list[str],
        destination: str,
        raw_destination: str,
    ) -> pd.DataFrame:
        numeric = [
            column
            for column in combined.select_dtypes(include="number").columns
            if column not in ("run_seed", "episodes")
        ]
        aggregate = combined.groupby(groups)[numeric].agg(["mean", "std"])
        aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
        aggregate = aggregate.reset_index()
        aggregate.to_csv(root / destination, index=False)
        combined.to_csv(root / raw_destination, index=False)
        return aggregate

    combined, paths = collect("seed_*/evaluation/summary.csv")
    if not paths:
        raise FileNotFoundError(f"No evaluation summaries under {root}")
    aggregate = aggregate_frame(
        combined,
        ["split", "method"],
        "aggregate_summary.csv",
        "all_seed_summaries.csv",
    )
    probe, probe_paths = collect("seed_*/evaluation/semantic_revision_probe.csv")
    if probe_paths:
        aggregate_frame(probe, ["features"], "aggregate_probe.csv", "all_seed_probes.csv")
    paired, paired_paths = collect("seed_*/evaluation/paired_comparisons.csv")
    if paired_paths:
        aggregate_frame(
            paired,
            ["split", "reference", "baseline", "metric"],
            "aggregate_paired_comparisons.csv",
            "all_seed_paired_comparisons.csv",
        )
    ablations, ablation_paths = collect("seed_*/evaluation/ablations/summary.csv")
    if ablation_paths:
        aggregate_frame(
            ablations,
            ["split", "variant", "parameter", "value", "method"],
            "aggregate_ablations.csv",
            "all_seed_ablations.csv",
        )
    return aggregate


def run_full(
    config: ExperimentConfig,
    skip_training: bool = False,
    reuse_trajectories: bool = False,
) -> Path:
    root = Path(config.train.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_config(config, root / "config.yaml")
    for seed in config.seeds:
        run_dir = root / f"seed_{seed}"
        checkpoint = run_dir / "best.pt"
        training_record = run_dir / "training.json"
        complete_training = False
        if checkpoint.exists() and training_record.exists():
            record = json.loads(training_record.read_text(encoding="utf-8"))
            complete_training = record.get("steps_completed", config.train.steps) == config.train.steps
        if not skip_training or not complete_training:
            checkpoint = train_model(config, seed)
        evaluate_checkpoint(config, checkpoint, reuse_trajectories=reuse_trajectories)
    aggregate_runs(root)
    return root / "aggregate_summary.csv"
