from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import MemoryConfig, load_config
from dreammem.diagnostics import (
    candidate_diagnostics,
    order_ablation,
    summarize_candidate_diagnostics,
    summarize_order_ablation,
)
from dreammem.experiment import load_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output")
    parser.add_argument("--shuffles", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = Path(args.evaluation)
    output = Path(args.output) if args.output else evaluation / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    raw_configs = json.loads((evaluation / "best_memory_configs.json").read_text(encoding="utf-8"))
    configs = {method: MemoryConfig(**values) for method, values in raw_configs.items()}
    trajectories = []
    for split in config.eval.splits:
        trajectories.extend(load_trajectories(evaluation / "trajectories" / f"{split}.pt"))
    candidates = candidate_diagnostics(trajectories, configs["evidence_guided"])
    candidate_summary = summarize_candidate_diagnostics(candidates)
    order = order_ablation(trajectories, configs, args.shuffles, config.eval.seed + 500)
    order_summary = summarize_order_ablation(order)
    candidates.to_csv(output / "candidate_episodes.csv", index=False)
    candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    order.to_csv(output / "order_episodes.csv", index=False)
    order_summary.to_csv(output / "order_summary.csv", index=False)
    print(candidate_summary.to_string(index=False))
    print(order_summary.to_string(index=False))


if __name__ == "__main__":
    main()
