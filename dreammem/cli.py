from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .ablation import run_ablations
from .config import load_config
from .experiment import evaluate_checkpoint, run_full
from .model import MaskedDiffusionTransformer
from .tasks import TaskGenerator
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dreammem",
        description="Semantic revision dynamics and trajectory-guided memory formation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "train", "evaluate", "ablate", "full"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="configs/full.yaml")
        if name in ("train", "full"):
            command.add_argument("--seed", type=int, action="append")
        if name in ("evaluate", "ablate"):
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output")
        if name == "evaluate":
            command.add_argument("--reuse-trajectories", action="store_true")
        if name == "full":
            command.add_argument("--skip-training", action="store_true")
            command.add_argument("--reuse-trajectories", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if getattr(args, "seed", None):
        config.seeds = args.seed
        config.train.seed = args.seed[0]
    if args.command == "inspect":
        from .task_factory import make_task_generator

        generator = make_task_generator(config)
        model = MaskedDiffusionTransformer(config.model)
        print(
            json.dumps(
                {
                    "task_type": config.task_type,
                    "reference_model_parameters": model.parameter_count(),
                    "concepts": {
                        split: len(concepts)
                        for split, concepts in generator.library.by_split.items()
                    },
                    "device": config.device,
                    "training_steps": config.train.steps,
                    "evaluation_episodes": config.eval.episodes,
                },
                indent=2,
            )
        )
    elif args.command == "train":
        for seed in config.seeds:
            print(train_model(config, seed))
    elif args.command == "evaluate":
        outputs = evaluate_checkpoint(
            config,
            args.checkpoint,
            args.output,
            args.reuse_trajectories,
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        print(pd.read_csv(outputs["summary"]).to_string(index=False))
    elif args.command == "ablate":
        summary = run_ablations(config, args.checkpoint, args.output)
        print(summary)
        print(pd.read_csv(summary).to_string(index=False))
    elif args.command == "full":
        summary = run_full(config, args.skip_training, args.reuse_trajectories)
        print(summary)
        print(pd.read_csv(summary).to_string(index=False))
