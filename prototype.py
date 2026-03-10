"""Phase 1: Single-file prototype proving the full chain.

Hardcoded expressions → Claude API → Qwen 3.5 (untrained LoRA) → output.
Validates connectivity and data flow without training.

Usage:
    python prototype.py
    # On H100 node (for local model):
    python prototype.py --with-local-model
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="API Adapter prototype")
    parser.add_argument(
        "--with-local-model",
        action="store_true",
        help="Also run through local Qwen model (requires GPU)",
    )
    args = parser.parse_args()

    from api_adapter.symbols import evaluate
    from api_adapter.api_client import get_client, query_claude, parse_answer

    # Test expressions: mix of standard and custom
    test_cases = [
        ("3 + 5", "standard"),
        ("12 * 4 - 7", "standard"),
        ("10 / 2 + 3", "standard"),
        ("3 θ 5", "custom"),          # 3 + 5 = 8
        ("12 γ 4 α 7", "custom"),     # 12 * 4 - 7 = 41
        ("10 β 2 θ 3", "custom"),     # 10 / 2 + 3 = 8
    ]

    client = get_client()

    print("=" * 60)
    print("API Adapter Prototype - End-to-End Chain Test")
    print("=" * 60)

    results = []
    for expr, expr_type in test_cases:
        correct = evaluate(expr)
        claude_response = query_claude(expr, client=client)
        claude_answer = parse_answer(claude_response)

        result = {
            "expression": expr,
            "type": expr_type,
            "correct": correct,
            "claude_response": claude_response,
            "claude_answer": claude_answer,
            "claude_correct": claude_answer == correct,
        }
        results.append(result)

        status = "OK" if result["claude_correct"] else "WRONG"
        print(f"\n[{status}] {expr} (type={expr_type})")
        print(f"  Correct answer: {correct}")
        print(f"  Claude says:    {claude_response} (parsed: {claude_answer})")

    # Summary
    correct_count = sum(1 for r in results if r["claude_correct"])
    print(f"\nClaude accuracy: {correct_count}/{len(results)}")

    if args.with_local_model:
        print("\n" + "=" * 60)
        print("Running through local adapter model (untrained)...")
        print("=" * 60)

        from api_adapter.local_model import load_model, format_adapter_prompt, generate

        model, tokenizer = load_model()

        prompts = [
            format_adapter_prompt(r["expression"], r["claude_response"], include_symbols=True)
            for r in results
        ]
        outputs = generate(model, tokenizer, prompts)

        for r, output in zip(results, outputs):
            print(f"\n  Expression: {r['expression']}")
            print(f"  Claude:     {r['claude_response']}")
            print(f"  Adapter:    {output}")
            print(f"  Correct:    {r['correct']}")

    print("\nPrototype complete.")


if __name__ == "__main__":
    main()
