from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config, save_config
from dreammem.constructive import evaluate_search
from dreammem.dcoder import DCoderAdapter, load_dcoder
from dreammem.evaluation import (
    bootstrap_intervals,
    concept_cluster_intervals,
    evaluate_trajectories,
    paired_comparisons,
    summarize_results,
)
from dreammem.experiment import generate_episodes, load_trajectories, save_trajectories
from dreammem.plots import plot_risk_coverage, plot_summary, plot_training
from dreammem.task_factory import make_task_components
from dreammem.trajectory import collect_episode_trajectories


@torch.no_grad()
def oracle_decoder(model, encoder, episodes, device, batch_size, support_count=6):
    rows = []
    for start in range(0, len(episodes), batch_size):
        subset = episodes[start:start + batch_size]
        batch = encoder.encode_batch(subset, support_count, device=device)
        inputs = batch.tokens.masked_fill(batch.query_mask, encoder.mask_token_id)
        ratios = torch.ones(len(subset), device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs, ratios)
        positions = encoder.query_label_positions
        binary = logits[:, positions][..., [encoder.no_token, encoder.yes_token]].float()
        targets = (batch.tokens[:, positions] == encoder.yes_token).long()
        loss = F.cross_entropy(binary.reshape(-1, 2), targets.reshape(-1), reduction="none")
        accuracy = (binary.argmax(-1) == targets).float().mean(dim=1)
        for offset, episode in enumerate(subset):
            rows.append(
                {
                    "split": episode.split,
                    "episode": episode.index,
                    "query_accuracy": float(accuracy[offset]),
                    "query_nll": float(loss.reshape(len(subset), -1)[offset].mean()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-seed", type=int, default=17100)
    parser.add_argument("--reuse-trajectories", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config.eval.seed = args.episode_seed
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    policy_dir = Path(args.policy)
    with (policy_dir / "policy.pkl").open("rb") as handle:
        policy = pickle.load(handle)
    search_config = json.loads((policy_dir / "best_search.json").read_text(encoding="utf-8"))
    baseline = Path(args.baseline_evaluation)
    raw_memory = json.loads((baseline / "best_memory_configs.json").read_text(encoding="utf-8"))
    memory_configs = {method: MemoryConfig(**values) for method, values in raw_memory.items()}
    memory = memory_configs["evidence_guided"]
    device = torch.device("cuda")
    tokenizer, raw_model = load_dcoder(args.model, device)
    model = DCoderAdapter(raw_model)
    generator, encoder = make_task_components(config, tokenizer, int(raw_model.generation_config.mask_token_id))
    all_results = []
    all_diagnostics = []
    oracle_rows = []
    timings = {}
    for split in config.eval.splits:
        path = output / "trajectories" / f"{split}.pt"
        if args.reuse_trajectories and path.exists():
            trajectories = load_trajectories(path)
            episodes = [trajectory.episode for trajectory in trajectories]
        else:
            episodes = generate_episodes(config, split)
            started = time.time()
            trajectories = collect_episode_trajectories(
                model,
                config,
                episodes,
                device,
                seed=config.eval.seed + sum(ord(character) for character in split),
                encoder=encoder,
                mask_token_id=encoder.mask_token_id,
                concept_codes_fn=encoder.concept_codes,
            )
            timings[split] = time.time() - started
            save_trajectories(path, trajectories)
        results = evaluate_trajectories(trajectories, memory_configs, generator.library)
        all_results.append(results)
        oracle_rows.extend(
            oracle_decoder(
                model,
                encoder,
                episodes,
                device,
                config.eval.batch_size,
                max(config.trajectory.support_stages),
            )
        )
        for strategy, budget in (
            ("closure_search", 0),
            ("random_search", search_config["best"]["random_search"]["budget"]),
            ("random_search_matched", search_config["best"]["learned_search"]["budget"]),
            ("learned_search", search_config["best"]["learned_search"]["budget"]),
        ):
            search_results, diagnostics = evaluate_search(
                trajectories,
                memory,
                generator.library,
                policy if strategy == "learned_search" else None,
                budget,
                strategy,
                config.eval.seed + 900 + budget,
            )
            all_results.append(search_results)
            all_diagnostics.append(diagnostics)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = pd.concat(all_results, ignore_index=True)
    metadata_path = Path(args.model) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if "seed" in metadata:
        combined["seed"] = int(metadata["seed"])
    else:
        combined["seed"] = int(pd.read_csv(baseline / "results_seen.csv", usecols=["seed"])["seed"].iloc[0])
    combined.to_csv(output / "results_all.csv", index=False)
    for split in config.eval.splits:
        combined[combined["split"] == split].to_csv(output / f"results_{split}.csv", index=False)
    summary = summarize_results(combined)
    intervals = bootstrap_intervals(combined, config.eval.bootstrap_samples, config.eval.seed + 82)
    clusters = concept_cluster_intervals(combined, config.eval.bootstrap_samples, config.eval.seed + 84)
    comparisons = paired_comparisons(
        combined,
        reference="learned_search",
        samples=config.eval.bootstrap_samples,
        seed=config.eval.seed + 83,
    )
    summary.to_csv(output / "summary.csv", index=False)
    intervals.to_csv(output / "intervals.csv", index=False)
    clusters.to_csv(output / "concept_cluster_intervals.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    pd.concat(all_diagnostics, ignore_index=True).to_csv(output / "search_diagnostics.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(output / "oracle_decoder.csv", index=False)
    plot_summary(summary, output / "metrics.png")
    plot_risk_coverage(combined, output / "risk_coverage.png")
    history_path = Path(args.model).parent / "history.csv"
    if history_path.exists():
        plot_training(history_path, output / "training.png")
    runtime = {
        "episode_seed": config.eval.seed,
        "episodes_per_split": config.eval.episodes,
        "forward_passes_per_episode": (
            config.trajectory.samples * config.trajectory.steps * len(config.trajectory.support_stages)
        ),
        "oracle_forward_passes_per_episode": 1,
        "trajectory_seconds": timings,
        "search": search_config,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(summary[["split", "method", "concept_accuracy", "query_accuracy", "query_nll"]].to_string(index=False))
    print(pd.DataFrame(oracle_rows).groupby("split")[["query_accuracy", "query_nll"]].mean().to_string())


if __name__ == "__main__":
    main()
