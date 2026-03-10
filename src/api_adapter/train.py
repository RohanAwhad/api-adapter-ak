"""GRPO training pipeline for the adapter model."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from api_adapter.local_model import format_adapter_prompt, load_model
from api_adapter.reward import correctness_reward, CORRECT_TOKEN


SYSTEM_PROMPT = (
    "You are an arithmetic verification assistant. "
    "Check the API model's answer and respond with only the correct number."
)

SYSTEM_PROMPT_WITH_CORRECT = (
    "You are an arithmetic verification assistant. "
    "Check the API model's answer. Respond with CORRECT if right, or the correct number if wrong."
)


def build_training_dataset(
    data_path: str | Path,
    include_symbols: bool = True,
    allow_correct_token: bool = False,
) -> tuple[Dataset, dict[str, int], dict[str, int | None]]:
    """Build a HuggingFace Dataset for GRPO training from baseline results.

    Uses conversational format (list of message dicts) as required by
    Unsloth's vLLM-backed GRPO pipeline.

    Returns:
        (dataset, prompt_key_to_answer, prompt_key_to_claude_answer)
        where prompt_key is the user message text for lookup.
    """
    samples = []
    prompt_key_to_answer: dict[str, int] = {}
    prompt_key_to_claude_answer: dict[str, int | None] = {}

    system_prompt = SYSTEM_PROMPT_WITH_CORRECT if allow_correct_token else SYSTEM_PROMPT

    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            user_msg = format_adapter_prompt(
                expression=item["expression"],
                claude_answer=item["claude_response"],
                include_symbols=include_symbols,
                allow_correct_token=allow_correct_token,
            )
            # Conversational format for TRL
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            samples.append({"prompt": prompt})
            prompt_key_to_answer[user_msg] = item["answer"]
            prompt_key_to_claude_answer[user_msg] = item.get("claude_answer")

    return Dataset.from_list(samples), prompt_key_to_answer, prompt_key_to_claude_answer


def make_reward_fn(
    prompt_key_to_answer: dict[str, int],
    prompt_key_to_claude_answer: dict[str, int | None],
):
    """Create a reward function compatible with TRL GRPOTrainer.

    In conversational mode, TRL calls:
        reward_func(prompts=list[list[dict]], completions=list[list[dict]], ...)
    """
    def correctness_reward_fn(prompts, completions, **kwargs) -> list[float]:
        # Extract user message text from conversational prompts for lookup
        user_msgs = [p[-1]["content"] for p in prompts]
        # Extract completion text
        completion_texts = [c[0]["content"] for c in completions]
        answers = [prompt_key_to_answer[m] for m in user_msgs]
        claude_answers = [prompt_key_to_claude_answer[m] for m in user_msgs]
        return correctness_reward(completion_texts, answers, claude_answers=claude_answers)

    return correctness_reward_fn


def train(
    data_path: str | Path,
    output_dir: str | Path = "outputs/grpo",
    include_symbols: bool = True,
    allow_correct_token: bool = False,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    num_generations: int = 4,
    model_name: str | None = None,
    lora_rank: int = 32,
    max_steps: int = -1,
):
    """Run GRPO training."""
    from api_adapter.local_model import DEFAULT_MODEL_NAME

    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name=model_name, lora_rank=lora_rank)

    print(f"Building dataset from: {data_path}")
    dataset, prompt_key_to_answer, prompt_key_to_claude_answer = build_training_dataset(
        data_path, include_symbols=include_symbols, allow_correct_token=allow_correct_token,
    )
    print(f"Training samples: {len(dataset)}")

    reward_fn = make_reward_fn(prompt_key_to_answer, prompt_key_to_claude_answer)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_completion_length=64,
        temperature=1.0,
        logging_steps=1,
        save_steps=200,
        bf16=True,
        report_to="none",
    )

    FastLanguageModel.for_training(model)

    # Workaround: TRL expects warnings_issued on the model but PEFT doesn't expose it
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_fn := reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir / "final_adapter")
    tokenizer.save_pretrained(output_dir / "final_adapter")
    print("Training complete.")
