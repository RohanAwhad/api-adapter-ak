"""Run GRPO training on the adapter model.

Usage (on H100 node):
    # Run all three conditions in parallel on separate GPUs:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/train_grpo.py --condition A &
    CUDA_VISIBLE_DEVICES=2,3 python scripts/train_grpo.py --condition B &
    CUDA_VISIBLE_DEVICES=4,5 python scripts/train_grpo.py --condition C &
"""

from __future__ import annotations

import unsloth  # noqa: F401 — must be imported before trl/transformers
import argparse

from api_adapter.train import train

CONDITIONS = {
    "A": {"include_symbols": True, "allow_correct_token": False},
    "B": {"include_symbols": False, "allow_correct_token": False},
    "C": {"include_symbols": True, "allow_correct_token": True},
    "D": {"include_symbols": True, "allow_correct_token": True, "vague_symbols": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=list(CONDITIONS.keys()), required=True,
        help="A = with symbols, B = without symbols, C = with symbols + CORRECT token, D = vague symbols + CORRECT token",
    )
    parser.add_argument("--data-path", default="data/baseline/train_baseline.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    cond = CONDITIONS[args.condition]
    output_dir = args.output_dir or f"outputs/grpo_condition_{args.condition}"

    print(f"Training Condition {args.condition}")
    print(f"  Include symbols:      {cond['include_symbols']}")
    print(f"  Allow CORRECT token:  {cond['allow_correct_token']}")
    print(f"  Vague symbols:        {cond.get('vague_symbols', False)}")
    print(f"  Output: {output_dir}")

    train(
        data_path=args.data_path,
        output_dir=output_dir,
        include_symbols=cond["include_symbols"],
        allow_correct_token=cond["allow_correct_token"],
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        lora_rank=args.lora_rank,
        vague_symbols=cond.get("vague_symbols", False),
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
