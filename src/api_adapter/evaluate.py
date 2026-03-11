"""Evaluation harness: compare API-only vs adapter conditions."""

from __future__ import annotations

import json
from pathlib import Path

from api_adapter.reward import extract_answer


def evaluate_predictions(
    predictions: list[dict],
    label: str = "",
) -> dict:
    """Compute accuracy metrics for a set of predictions.

    Args:
        predictions: List of dicts with "answer" (ground truth) and "predicted" fields.
        label: Label for this evaluation condition.

    Returns:
        Dict with overall and per-type accuracy.
    """
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("predicted") == p["answer"])

    custom = [p for p in predictions if p["type"] == "custom"]
    standard = [p for p in predictions if p["type"] == "standard"]

    custom_correct = sum(1 for p in custom if p.get("predicted") == p["answer"])
    standard_correct = sum(1 for p in standard if p.get("predicted") == p["answer"])

    result = {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "custom_total": len(custom),
        "custom_correct": custom_correct,
        "custom_accuracy": custom_correct / len(custom) if custom else 0,
        "standard_total": len(standard),
        "standard_correct": standard_correct,
        "standard_accuracy": standard_correct / len(standard) if standard else 0,
    }

    if label:
        print(f"\n=== {label} ===")
    print(f"Overall:  {correct}/{total} = {result['accuracy']:.1%}")
    print(f"Custom:   {custom_correct}/{len(custom)} = {result['custom_accuracy']:.1%}")
    print(f"Standard: {standard_correct}/{len(standard)} = {result['standard_accuracy']:.1%}")

    return result


def evaluate_claude_baseline(results_path: str | Path) -> dict:
    """Evaluate Claude-only baseline from saved results."""
    with open(results_path) as f:
        results = [json.loads(line) for line in f]

    predictions = []
    for r in results:
        predictions.append({
            "answer": r["answer"],
            "predicted": r["claude_answer"],
            "type": r["type"],
        })

    return evaluate_predictions(predictions, label="Claude API Only")


def evaluate_adapter(
    results_path: str | Path,
    model,
    tokenizer,
    include_symbols: bool = True,
    label: str = "Adapter",
) -> dict:
    """Evaluate adapter model on test set.

    Args:
        results_path: Path to JSONL with Claude baseline results for test set.
        model: Loaded adapter model.
        tokenizer: Tokenizer.
        include_symbols: Whether to include symbol definitions in prompt.
        label: Label for this condition.

    Returns:
        Metrics dict.
    """
    from api_adapter.local_model import format_adapter_prompt, generate

    with open(results_path) as f:
        results = [json.loads(line) for line in f]

    prompts = []
    for r in results:
        claude_ans = r.get("claude_answer")
        claude_ans_str = str(claude_ans) if claude_ans is not None else "none"
        prompt = format_adapter_prompt(
            expression=r["expression"],
            claude_answer=claude_ans_str,
            include_symbols=include_symbols,
        )
        prompts.append(prompt)

    print(f"Running inference on {len(prompts)} samples...")
    outputs = generate(model, tokenizer, prompts)

    predictions = []
    for r, output in zip(results, outputs):
        predicted = extract_answer(output)
        predictions.append({
            "answer": r["answer"],
            "predicted": predicted,
            "type": r["type"],
            "output": output,
        })

    return evaluate_predictions(predictions, label=label)


def compare_conditions(metrics_list: list[dict]) -> None:
    """Print a comparison table of all conditions."""
    print("\n" + "=" * 70)
    print(f"{'Condition':<30} {'Overall':>10} {'Custom':>10} {'Standard':>10}")
    print("-" * 70)
    for m in metrics_list:
        print(
            f"{m['label']:<30} "
            f"{m['accuracy']:>9.1%} "
            f"{m['custom_accuracy']:>9.1%} "
            f"{m['standard_accuracy']:>9.1%}"
        )
    print("=" * 70)
