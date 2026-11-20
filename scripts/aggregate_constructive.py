from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.experiment import aggregate_runs


def aggregate_auxiliary(root: Path, pattern: str, groups: list[str], destination: str) -> None:
    frames = []
    for path in root.glob(pattern):
        frame = pd.read_csv(path)
        seed = next(part for part in path.parts if part.startswith("seed_"))
        frame["run_seed"] = int(seed.split("_")[-1])
        frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    numeric = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column not in ("run_seed", "episode")
    ]
    per_seed = combined.groupby([*groups, "run_seed"])[numeric].mean().reset_index()
    aggregate = per_seed.groupby(groups)[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    aggregate.reset_index().to_csv(root / destination, index=False)
    combined.to_csv(root / f"all_seed_{destination}", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    summary = aggregate_runs(root)
    aggregate_auxiliary(
        root,
        "seed_*/evaluation/search_diagnostics.csv",
        ["split", "method"],
        "aggregate_search_diagnostics.csv",
    )
    aggregate_auxiliary(
        root,
        "seed_*/evaluation/oracle_decoder.csv",
        ["split"],
        "aggregate_oracle_decoder.csv",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
