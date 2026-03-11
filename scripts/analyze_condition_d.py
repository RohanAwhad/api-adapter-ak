"""Analyze Condition D results: evaluate on test set, plot rewards, extract examples."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_training_rewards(log_paths: list[str]) -> list[dict]:
    """Parse reward and completion length from training logs."""
    entries = []
    for log_path in log_paths:
        with open(log_path) as f:
            text = f.read()
        rewards = [float(m) for m in re.findall(r"rewards/correctness_reward_fn/mean.: ([0-9.]+)", text)]
        lengths = [float(m) for m in re.findall(r"completions/mean_length.: ([0-9.]+)", text)]
        clipped = [float(m) for m in re.findall(r"completions/clipped_ratio.: ([0-9.]+)", text)]
        for r, l, c in zip(rewards, lengths, clipped):
            entries.append({"reward": r, "mean_length": l, "clipped_ratio": c})
    return entries


def plot_rewards(entries: list[dict], output_path: str):
    """Plot reward progression split by batch type, plus rolling average."""
    custom_idx, custom_r = [], []
    std_idx, std_r = [], []
    for i, e in enumerate(entries):
        if e["mean_length"] >= 150:
            custom_idx.append(i)
            custom_r.append(e["reward"])
        else:
            std_idx.append(i)
            std_r.append(e["reward"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Condition D: GRPO Training with Vague Symbol Definitions", fontsize=14, fontweight="bold")

    # Top-left: All rewards
    ax = axes[0, 0]
    ax.scatter(custom_idx, custom_r, c="tab:red", s=20, alpha=0.7, label="Custom symbols")
    ax.scatter(std_idx, std_r, c="tab:blue", s=20, alpha=0.7, label="Standard arithmetic")
    ax.set_xlabel("Generation batch")
    ax.set_ylabel("Reward")
    ax.set_title("All Batch Rewards")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Custom symbol rewards with rolling average
    ax = axes[0, 1]
    ax.scatter(range(len(custom_r)), custom_r, c="tab:red", s=25, alpha=0.5, label="Per-batch")
    if len(custom_r) >= 5:
        window = 5
        rolling = [np.mean(custom_r[max(0, i - window + 1):i + 1]) for i in range(len(custom_r))]
        ax.plot(range(len(rolling)), rolling, c="darkred", linewidth=2.5, label=f"Rolling avg (w={window})")
    ax.set_xlabel("Custom symbol batch #")
    ax.set_ylabel("Reward")
    ax.set_title("Custom Symbol Reward Progression")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Custom reward distribution early vs late
    ax = axes[1, 0]
    mid = len(custom_r) // 2
    early = custom_r[:mid] if mid > 0 else custom_r
    late = custom_r[mid:] if mid > 0 else []
    bins = np.arange(0, 1.15, 0.1)
    if early:
        ax.hist(early, bins=bins, alpha=0.6, color="tab:orange", label=f"First half (n={len(early)}, avg={np.mean(early):.2f})")
    if late:
        ax.hist(late, bins=bins, alpha=0.6, color="tab:green", label=f"Second half (n={len(late)}, avg={np.mean(late):.2f})")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Custom Symbol Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Completion length over time
    ax = axes[1, 1]
    custom_lens = [e["mean_length"] for e in entries if e["mean_length"] >= 150]
    ax.scatter(range(len(custom_lens)), custom_lens, c="tab:red", s=20, alpha=0.7, label="Custom")
    ax.set_xlabel("Custom symbol batch #")
    ax.set_ylabel("Mean completion length")
    ax.set_title("Custom Symbol Completion Length")
    ax.axhline(y=512, color="gray", linestyle="--", alpha=0.5, label="Max (512)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward plot to {output_path}")


def run_evaluation(adapter_path: str, test_data_path: str):
    """Run model evaluation on the test set using vLLM for fast inference."""
    import unsloth  # noqa
    from unsloth import FastLanguageModel
    from vllm import SamplingParams
    from api_adapter.local_model import format_adapter_prompt
    from api_adapter.reward import extract_answer
    from api_adapter.evaluate import evaluate_predictions

    with open(test_data_path) as f:
        test_items = [json.loads(line) for line in f]

    print(f"\nLoading adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )
    FastLanguageModel.for_inference(model)

    # Build prompts with vague symbols (matching Condition D)
    messages_list = []
    system_prompt = (
        "You are an adapter that checks an API model's arithmetic answer. "
        "If the API answer is correct, respond with \\boxed{CORRECT}. "
        "If wrong or missing, compute the correct answer and respond with \\boxed{answer}. "
        "Use the symbol definitions to evaluate custom expressions. Be concise."
    )
    for item in test_items:
        ca = item.get("claude_answer")
        ca_str = str(ca) if ca is not None else "none"
        user_msg = format_adapter_prompt(
            expression=item["expression"],
            claude_answer=ca_str,
            include_symbols=True,
            allow_correct_token=True,
            vague_symbols=True,
        )
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ])

    # Tokenize all prompts
    print(f"Running vLLM inference on {len(messages_list)} test samples...")
    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in messages_list
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    outputs = model.fast_generate(
        texts,
        sampling_params=sampling_params,
        lora_request=model.load_lora("outputs/grpo_condition_D/final_adapter"),
    )

    # Extract text from outputs
    all_outputs = [o.outputs[0].text for o in outputs]
    print(f"Inference complete.")

    # Evaluate
    predictions = []
    for item, output in zip(test_items, all_outputs):
        ca = item.get("claude_answer")
        predicted = extract_answer(output, claude_answer=ca)
        predictions.append({
            "answer": item["answer"],
            "predicted": predicted,
            "type": item["type"],
            "output": output,
            "expression": item["expression"],
            "claude_answer": ca,
        })

    metrics = evaluate_predictions(predictions, label="Condition D (Vague Symbols)")

    # Print examples where model got custom symbols correct
    custom_correct = [p for p in predictions if p["type"] == "custom" and p["predicted"] == p["answer"]]
    custom_wrong = [p for p in predictions if p["type"] == "custom" and p["predicted"] != p["answer"]]

    print(f"\n{'='*70}")
    print(f"EXAMPLES: Custom symbol problems SOLVED by adapter ({len(custom_correct)} total)")
    print(f"{'='*70}")
    for i, p in enumerate(custom_correct[:15]):
        ca_str = str(p["claude_answer"]) if p["claude_answer"] is not None else "none"
        claude_status = "correct" if p["claude_answer"] == p["answer"] else "WRONG" if p["claude_answer"] is not None else "MISSING"
        print(f"\n  [{i+1}] {p['expression']}")
        print(f"      True: {p['answer']} | Claude: {ca_str} ({claude_status}) | Adapter: {p['predicted']}")
        print(f"      Output: {p['output'][:300]}")

    print(f"\n{'='*70}")
    print(f"EXAMPLES: Custom symbol problems FAILED ({len(custom_wrong)} total)")
    print(f"{'='*70}")
    for i, p in enumerate(custom_wrong[:5]):
        ca_str = str(p["claude_answer"]) if p["claude_answer"] is not None else "none"
        print(f"\n  [{i+1}] {p['expression']}")
        print(f"      True: {p['answer']} | Claude: {ca_str} | Adapter: {p['predicted']}")
        print(f"      Output: {p['output'][:300]}")

    # Save predictions
    pred_path = "outputs/grpo_condition_D/test_predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved predictions to {pred_path}")

    return metrics


def main():
    log_paths = [
        "outputs/grpo_condition_D.log",
        "outputs/grpo_condition_D_resumed.log",
    ]
    test_data_path = "data/baseline/test_baseline.jsonl"
    adapter_path = "outputs/grpo_condition_D/final_adapter"
    plot_path = "outputs/grpo_condition_D/reward_plot.png"

    # 1. Plot training rewards
    print("Parsing training logs...")
    entries = parse_training_rewards(log_paths)
    print(f"Total generation batches: {len(entries)}")
    plot_rewards(entries, plot_path)

    # 2. Run evaluation on test set
    if Path(adapter_path).exists():
        metrics = run_evaluation(adapter_path, test_data_path)
    else:
        print(f"\nAdapter not found at {adapter_path}, skipping evaluation.")


if __name__ == "__main__":
    main()
