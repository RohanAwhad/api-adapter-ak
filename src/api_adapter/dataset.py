"""Dataset generation for arithmetic expressions."""

from __future__ import annotations

import json
import random
from pathlib import Path

from api_adapter.symbols import generate_expression


def generate_dataset(
    n_custom: int = 1000,
    n_standard: int = 1000,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> dict[str, list[dict]]:
    """Generate arithmetic dataset with train/test split.

    Args:
        n_custom: Number of custom-symbol expressions.
        n_standard: Number of standard arithmetic expressions.
        seed: Random seed.
        train_ratio: Fraction for training set.

    Returns:
        {"train": [...], "test": [...]} where each item is
        {"expression": str, "answer": int, "type": "custom"|"standard"}
    """
    rng = random.Random(seed)

    custom_samples = []
    for _ in range(n_custom):
        num_ops = rng.randint(2, 4)
        expr, answer = generate_expression(num_ops, use_custom=True, rng=rng)
        custom_samples.append({"expression": expr, "answer": answer, "type": "custom"})

    standard_samples = []
    for _ in range(n_standard):
        num_ops = rng.randint(2, 4)
        expr, answer = generate_expression(num_ops, use_custom=False, rng=rng)
        standard_samples.append({"expression": expr, "answer": answer, "type": "standard"})

    # Stratified split
    rng.shuffle(custom_samples)
    rng.shuffle(standard_samples)

    n_custom_train = int(len(custom_samples) * train_ratio)
    n_standard_train = int(len(standard_samples) * train_ratio)

    train = custom_samples[:n_custom_train] + standard_samples[:n_standard_train]
    test = custom_samples[n_custom_train:] + standard_samples[n_standard_train:]

    rng.shuffle(train)
    rng.shuffle(test)

    return {"train": train, "test": test}


def save_dataset(dataset: dict[str, list[dict]], output_dir: str | Path) -> None:
    """Save dataset splits to JSONL files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in dataset.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved {len(samples)} samples to {path}")


def load_dataset(data_dir: str | Path) -> dict[str, list[dict]]:
    """Load dataset from JSONL files."""
    data_dir = Path(data_dir)
    result = {}
    for split in ["train", "test"]:
        path = data_dir / f"{split}.jsonl"
        with open(path) as f:
            result[split] = [json.loads(line) for line in f]
    return result
