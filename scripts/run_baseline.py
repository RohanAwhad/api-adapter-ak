"""Run Claude baseline evaluation on the dataset.

Sends all expressions through Claude and saves results with Claude's answers.
These results are needed as training input for the adapter model.

Usage:
    python scripts/run_baseline.py [--split train test] [--concurrency 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api_adapter.api_client import run_baseline
from api_adapter.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train", "test"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/baseline")
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.split:
        print(f"\n{'='*60}")
        print(f"Running baseline on {split} split ({len(dataset[split])} samples)")
        print(f"{'='*60}")

        results = run_baseline(dataset[split], concurrency=args.concurrency)

        output_path = output_dir / f"{split}_baseline.jsonl"
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
