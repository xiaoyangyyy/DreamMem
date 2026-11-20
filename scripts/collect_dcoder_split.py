from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import load_config, save_config
from dreammem.dcoder import DCoderAdapter, load_dcoder
from dreammem.experiment import save_trajectories
from dreammem.task_factory import make_task_components
from dreammem.trajectory import collect_episode_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    count = args.episodes or config.eval.episodes
    seed = args.seed or config.eval.seed
    device = torch.device("cuda")
    tokenizer, model = load_dcoder(args.model, device)
    adapter = DCoderAdapter(model)
    generator, encoder = make_task_components(config, tokenizer, int(model.generation_config.mask_token_id))
    episodes = [generator.sample(args.split, index, seed + index) for index in range(count)]
    started = time.time()
    trajectories = collect_episode_trajectories(
        adapter,
        config,
        episodes,
        device,
        seed=seed + sum(ord(character) for character in args.split),
        encoder=encoder,
        mask_token_id=encoder.mask_token_id,
        concept_codes_fn=encoder.concept_codes,
    )
    output = Path(args.output)
    save_trajectories(output, trajectories)
    save_config(config, output.parent / f"{args.split}_config.yaml")
    (output.parent / f"{args.split}_runtime.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "episodes": count,
                "episode_seed": seed,
                "seconds": time.time() - started,
                "forward_passes_per_episode": (
                    config.trajectory.samples
                    * config.trajectory.steps
                    * len(config.trajectory.support_stages)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
