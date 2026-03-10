"""Local model (Qwen 3 + LoRA) loading and inference via Unsloth."""

from __future__ import annotations

from unsloth import FastLanguageModel


DEFAULT_MODEL_NAME = "unsloth/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_RANK = 32


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_rank: int = DEFAULT_LORA_RANK,
    fast_inference: bool = True,
    gpu_memory_utilization: float = 0.6,
):
    """Load Qwen 3 with LoRA adapter via Unsloth.

    Args:
        model_name: HuggingFace model identifier.
        max_seq_length: Maximum sequence length.
        lora_rank: LoRA rank.
        fast_inference: Enable vLLM fast inference (required for GRPO).
        gpu_memory_utilization: GPU memory fraction for vLLM.

    Returns:
        (model, tokenizer) tuple.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 on H100)
        load_in_4bit=False,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


def format_adapter_prompt(
    expression: str,
    claude_answer: str,
    include_symbols: bool = True,
    allow_correct_token: bool = False,
) -> str:
    """Format the input prompt for the adapter model.

    Args:
        expression: The arithmetic expression.
        claude_answer: Claude's response to the expression.
        include_symbols: Whether to include custom symbol definitions.
        allow_correct_token: If True, instruct the model it can emit CORRECT
            to accept Claude's answer as-is (Condition C).

    Returns:
        Formatted prompt string.
    """
    from api_adapter.symbols import SYMBOL_DEFINITIONS

    parts = []
    if include_symbols:
        parts.append(SYMBOL_DEFINITIONS)
        parts.append("")

    parts.append(f"Expression: {expression}")
    parts.append(f"API model answer: {claude_answer}")
    parts.append("")

    if allow_correct_token:
        parts.append(
            "Verify the API model's answer. "
            "If correct, respond with CORRECT. "
            "If incorrect, respond with the correct number. "
            "Respond with ONLY CORRECT or a number."
        )
    else:
        parts.append(
            "Verify the API model's answer. If correct, repeat it. "
            "If incorrect, provide the correct answer. "
            "Respond with ONLY the number."
        )
    return "\n".join(parts)


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 32,
) -> list[str]:
    """Generate responses from the local model."""
    FastLanguageModel.for_inference(model)

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)

    return outputs
