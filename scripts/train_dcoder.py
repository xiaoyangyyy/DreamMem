from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dreammem.config import load_config
from dreammem.dcoder_train import train_dcoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcoder.yaml")
    parser.add_argument("--model", default="models/open-dcoder-0.5B")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(train_dcoder(load_config(args.config), args.model, args.seed))


if __name__ == "__main__":
    main()
