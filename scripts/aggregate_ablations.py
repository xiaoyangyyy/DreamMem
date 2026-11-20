from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate(root: Path, filename: str, destination: str) -> None:
    frames = []
    for path in root.glob(f"seed_*/evaluation/ablations/{filename}"):
        frame = pd.read_csv(path)
        seed = next(part for part in path.parts if part.startswith("seed_"))
        frame["run_seed"] = int(seed.split("_")[-1])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(root / f"all_seed_{destination}", index=False)
    numeric = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in ("run_seed", "episode", "seed")
    ]
    per_seed = combined.groupby(["split", "method", "run_seed"])[numeric].mean().reset_index()
    aggregate = per_seed.groupby(["split", "method"])[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{column}_{statistic}" for column, statistic in aggregate.columns]
    aggregate.reset_index().to_csv(root / destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    aggregate(root, "ablation_results.csv", "aggregate_ablation_results.csv")
    aggregate(root, "ablation_costs.csv", "aggregate_ablation_costs.csv")
    print(pd.read_csv(root / "aggregate_ablation_results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
