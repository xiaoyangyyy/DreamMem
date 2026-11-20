from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import load_config
from dreammem.dcoder_experiment import evaluate_dcoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcoder.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-trajectories", action="store_true")
    args = parser.parse_args()
    outputs = evaluate_dcoder(
        load_config(args.config),
        args.model,
        args.output,
        args.reuse_trajectories,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print(pd.read_csv(outputs["summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
