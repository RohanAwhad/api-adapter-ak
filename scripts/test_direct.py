"""Test with vs without LoRA using exact training prompt format."""

from unsloth import FastLanguageModel
from vllm import SamplingParams
from api_adapter.local_model import format_adapter_prompt
from api_adapter.reward import extract_answer


SYSTEM_PROMPT = (
    "You are an adapter that checks an API model's arithmetic answer. "
    "If the API answer is correct, respond with \\boxed{CORRECT}. "
    "If wrong or missing, compute the correct answer and respond with \\boxed{answer}. "
    "Use the symbol definitions to evaluate custom expressions. Be concise."
)

# (expression, claude_answer, expected_answer)
QUERIES = [
    ("5 θ 3", "none", 8),
    ("10 α 4", "none", 6),
    ("7 γ 6", "none", 42),
    ("20 β 5", "none", 4),
    ("3 θ 4 γ 5", "none", 23),
    ("100 α 30 α 20", "none", 50),
    ("12 γ 3 θ 8", "none", 44),
    ("6 γ 7 θ 2 α 10", "none", 34),
    ("100 β 4 θ 3 γ 5", "none", 40),
    ("15 θ 25", "100", 40),
    ("9 γ 9", "18", 81),
    ("50 β 10", "500", 5),
    ("80 α 35", "115", 45),
    ("4 θ 6", "10", 10),
    ("8 γ 7", "56", 56),
]


def build_prompts(tokenizer):
    """Build prompts using exact training format."""
    messages_list = []
    for expr, api_ans, _ in QUERIES:
        user_msg = format_adapter_prompt(
            expression=expr,
            claude_answer=api_ans,
            include_symbols=True,
            allow_correct_token=True,
            vague_symbols=True,
        )
        messages_list.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

    return [
        tokenizer.apply_chat_template(
            msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        for msgs in messages_list
    ]


def run_and_print(label, outputs):
    correct_count = 0
    results = []
    for (expr, api_ans, expected), out in zip(QUERIES, outputs):
        text = out.outputs[0].text.strip()
        ca = int(api_ans) if api_ans != "none" else None
        predicted = extract_answer(text, claude_answer=ca)
        is_correct = predicted == expected
        if is_correct:
            correct_count += 1
        results.append((expr, api_ans, expected, predicted, is_correct, text))

    total = len(QUERIES)
    print(f"\n{'=' * 70}")
    print(f"{label}: {correct_count}/{total} = {100*correct_count/total:.1f}%")
    print(f"{'=' * 70}")
    for expr, api_ans, expected, predicted, is_correct, text in results:
        status = "OK" if is_correct else "WRONG"
        print(f"  [{status:5s}] {expr:30s} | API={api_ans:5s} | exp={expected:10} | got={predicted}")
    return results


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/grpo_condition_D/final_adapter",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        fast_inference=True,
        gpu_memory_utilization=0.5,
    )
    FastLanguageModel.for_inference(model)

    texts = build_prompts(tokenizer)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # Run WITH LoRA
    lora_req = model.load_lora("outputs/grpo_condition_D/final_adapter")
    outputs_lora = model.fast_generate(texts, sampling_params, lora_request=lora_req)
    results_lora = run_and_print("WITH LoRA (trained)", outputs_lora)

    # Run WITHOUT LoRA (base Qwen3-8B)
    outputs_base = model.fast_generate(texts, sampling_params)
    results_base = run_and_print("WITHOUT LoRA (base Qwen3-8B)", outputs_base)

    # Side-by-side comparison
    print(f"\n{'=' * 70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Expression':30s} | {'API':5s} | {'Expected':>8s} | {'Base':>8s} | {'LoRA':>8s} | Delta")
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+------")
    base_correct = 0
    lora_correct = 0
    for rb, rl in zip(results_base, results_lora):
        expr, api_ans, expected, pred_b, ok_b, _ = rb
        _, _, _, pred_l, ok_l, _ = rl
        base_correct += ok_b
        lora_correct += ok_l
        b_str = str(pred_b) if pred_b is not None else "None"
        l_str = str(pred_l) if pred_l is not None else "None"
        if ok_l and not ok_b:
            delta = " +LoRA"
        elif ok_b and not ok_l:
            delta = " -LoRA"
        elif ok_l and ok_b:
            delta = "  both"
        else:
            delta = "  none"
        print(f"  {expr:30s} | {api_ans:5s} | {expected:>8} | {b_str:>8s} | {l_str:>8s} |{delta}")

    total = len(QUERIES)
    print(f"\n  Base: {base_correct}/{total} = {100*base_correct/total:.1f}%")
    print(f"  LoRA: {lora_correct}/{total} = {100*lora_correct/total:.1f}%")


if __name__ == "__main__":
    main()
