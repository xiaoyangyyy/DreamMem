from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import load_config, save_config
from dreammem.dcoder import DCoderAdapter, load_dcoder
from dreammem.evaluation import tune_memory_configs
from dreammem.experiment import generate_episodes, load_trajectories, save_trajectories
from dreammem.task_factory import make_task_components
from dreammem.trajectory import collect_episode_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-trajectories", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    device = torch.device("cuda")
    tokenizer, raw_model = load_dcoder(args.model, device)
    model = DCoderAdapter(raw_model)
    generator, encoder = make_task_components(config, tokenizer, int(raw_model.generation_config.mask_token_id))
    trajectories = {}
    timings = {}
    for split in ("val", "val_compositional"):
        path = output / "trajectories" / f"{split}.pt"
        if args.reuse_trajectories and path.exists():
            current = load_trajectories(path)
        else:
            episodes = generate_episodes(config, split)
            started = time.time()
            current = collect_episode_trajectories(
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
            save_trajectories(path, current)
        trajectories[split] = current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tuning, history = tune_memory_configs(
        trajectories["val"],
        config.memory,
        generator.library,
        config.eval.tuning_trials,
        config.eval.seed + 81,
        trajectories["val_compositional"],
    )
    history.to_csv(output / "tuning_history.csv", index=False)
    (output / "best_memory_configs.json").write_text(
        json.dumps({method: asdict(value) for method, value in tuning.items()}, indent=2),
        encoding="utf-8",
    )
    metadata_path = Path(args.model) / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    (output / "runtime.json").write_text(
        json.dumps({"trajectory_seconds": timings, "seed": metadata.get("seed", config.train.seed)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
