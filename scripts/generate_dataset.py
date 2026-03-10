"""Generate the arithmetic dataset (2000 expressions, train/test split)."""

from api_adapter.dataset import generate_dataset, save_dataset


def main():
    print("Generating dataset...")
    dataset = generate_dataset(n_custom=1000, n_standard=1000, seed=42)
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Test:  {len(dataset['test'])} samples")

    save_dataset(dataset, "data")
    print("Done.")


if __name__ == "__main__":
    main()
