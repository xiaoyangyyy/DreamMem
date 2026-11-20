from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config, save_config
from dreammem.dcoder import DCoderAdapter, load_dcoder
from dreammem.decoders import collect_ar_trajectories, collect_mlm_trajectories
from dreammem.evaluation import evaluate_trajectories, summarize_results
from dreammem.experiment import generate_episodes, load_trajectories, save_trajectories
from dreammem.task_factory import make_task_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["ar", "mlm"], required=True)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument("--reuse-trajectories", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.episode_seed is not None:
        config.eval.seed = args.episode_seed
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    raw = json.loads((Path(args.baseline_evaluation) / "best_memory_configs.json").read_text(encoding="utf-8"))
    configs = {method: MemoryConfig(**values) for method, values in raw.items()}
    device = torch.device("cuda")
    tokenizer, raw_model = load_dcoder(args.model, device)
    model = DCoderAdapter(raw_model, causal=args.mode == "ar")
    generator, encoder = make_task_components(config, tokenizer, int(raw_model.generation_config.mask_token_id))
    all_results = []
    timings = {}
    for split in config.eval.splits:
        path = output / "trajectories" / f"{args.mode}_{split}.pt"
        if args.reuse_trajectories and path.exists():
            trajectories = load_trajectories(path)
        else:
            episodes = generate_episodes(config, split)
            started = time.time()
            collector = collect_ar_trajectories if args.mode == "ar" else collect_mlm_trajectories
            trajectories = collector(
                model,
                config,
                episodes,
                device,
                encoder,
                config.eval.seed + sum(ord(character) for character in split),
            )
            timings[split] = time.time() - started
            save_trajectories(path, trajectories)
        results = evaluate_trajectories(trajectories, configs, generator.library)
        results["decoder"] = args.mode
        all_results.append(results)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output / "results_all.csv", index=False)
    summary = combined.groupby(["decoder", "split", "method"])[
        ["concept_accuracy", "query_accuracy", "query_nll", "brier", "confidence", "validation_error"]
    ].mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    positions = 3 + config.data.n_queries
    forward_passes = (
        14 * len(config.trajectory.support_stages) * positions
        if args.mode == "ar"
        else 6 * len(config.trajectory.support_stages)
    )
    metadata_path = Path(args.model) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "decoder": args.mode,
                "seed": metadata.get("seed", config.train.seed),
                "episode_seed": config.eval.seed,
                "forward_passes_per_episode": forward_passes,
                "positions_per_trajectory": positions,
                "trajectory_seconds": timings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
