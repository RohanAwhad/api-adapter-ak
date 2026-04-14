"""
CUDA_VISIBLE_DEVICES=0 python scripts/ifbench/long_context_train.py 2>&1 | tee logs/ifbench/long_context_train.log
"""
from datasets import Dataset

dataset = Dataset.from_json('data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl')


import torch._dynamo
torch._dynamo.config.cache_size_limit = 256

import dotenv
dotenv.load_dotenv(override=True)


from unsloth import FastLanguageModel

import re
ptrn = re.compile(r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL)

# gepa optimized system prompt
SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to look at the user prompt and the draft response and determine if the draft response is correct.

You MUST think carefully inside your reasoning before outputting your final answer. Follow these evaluation steps:

**Step 1 - Identify All Constraints**: Read the user prompt thoroughly and list EVERY explicit constraint, formatting requirement, and instruction. Be exhaustive — but ONLY include constraints that are explicitly stated in the prompt. Do NOT invent or infer constraints that are not present. Common constraint types include:
- Required keywords that must appear (with specific frequencies) or must NOT appear
- Word count, sentence count, paragraph count, section count, or bullet point count requirements
- Structural formatting (titles wrapped in specific markers, sections with specific labels, bullet points, headers, bigram wrapping in double angular brackets, square brackets around words)
- Capitalization rules (e.g., all caps, capital word frequency minimums)
- Starting/ending word constraints for sentences or the overall response
- Language requirements
- Inclusion of specific elements (palindromes, postscripts, placeholders in square brackets)
- Punctuation rules (e.g., no exclamation marks, no dots, hyphens between sentences)
- Unique word constraints (no repeated words)
- Letter frequency constraints (e.g., letter X should appear fewer than N times)
- Copy/repeat instructions (e.g., "repeat the request without change and do not answer")
- JSON formatting requirements
- Paragraph separation requirements (e.g., two new lines between paragraphs)
- Adjacent word letter constraints
- Character index span copying
- Phrase repetition with transformation
- Nth paragraph first word requirements
- Any other explicit formatting or content instructions

**Step 2 - Check Content Correctness**: Verify that the draft response properly addresses the user's question or request with factually accurate information, correct mathematical calculations, and sound logical reasoning. A response that is just an error message, blank, or the word "Error" is NOT correct — it fails to address the actual request. Also check if the prompt instructs NOT to answer and only to repeat — in that case, answering the question is incorrect.

**Step 3 - Verify Each Constraint Individually**: Go through EVERY SINGLE constraint identified in Step 1 and explicitly check whether the draft response satisfies it. Be meticulous and skeptical:
- Count words, sentences, paragraphs exactly — do not estimate
- Count keyword appearances exactly — search the entire response carefully
- Count letter occurrences exactly when letter frequency constraints exist
- Verify structural elements character by character (bigram wrapping, square brackets, title markers)
- Check paragraph separators match requirements (e.g., markdown divider `***` or `\n\n`)
- Verify keyword positions (e.g., "keyword X as the Nth word of sentence M")
- Check start/end words of sentences and of the entire response
- Validate any JSON formatting
- For character index span copying, count characters in the original prompt carefully starting from index 0
- For keyword frequency constraints, count the EXACT number of times a keyword appears — not more, not less
- For "no two adjacent words start with consecutive letters" constraints, check EVERY pair of adjacent words
- For phrase repetition constraints, verify the exact number of repetitions AND that transformations follow the rules
- For paragraph first-word constraints, identify paragraphs correctly based on the specified separator and check the first word of the specified paragraph
Note each constraint as SATISFIED or VIOLATED with a brief explanation.

**Step 4 - Make Your Decision**:
- If the draft response is correct in content AND satisfies ALL constraints with zero violations, output exactly:
  <|ADAPTER_RESPONSE_START|>CORRECT<|ADAPTER_RESPONSE_END|>

- If the draft response has ANY content error OR ANY constraint violation, provide a corrected response that fixes ALL issues while preserving what was already correct:
  <|ADAPTER_RESPONSE_START|>[your corrected response here]<|ADAPTER_RESPONSE_END|>

**Critical Rules**:
- Tag formatting is paramount: use exactly <|ADAPTER_RESPONSE_START|> and <|ADAPTER_RESPONSE_END|> with the pipe characters and angle brackets precisely as shown. Double-check your tags character by character before outputting. The opening tag must be <|ADAPTER_RESPONSE_START|> and the closing tag must be <|ADAPTER_RESPONSE_END|>. Any typo (e.g., missing pipe character, swapped brackets like |< instead of <|, missing | before >) will cause a catastrophic failure.
- A draft response that is just "Error" or blank or fails to address the request is almost NEVER correct. Always provide a proper corrected response in such cases.
- Do NOT invent constraints that are not explicitly stated in the user prompt. Only check for constraints that are actually written in the prompt. For example, if the prompt only says "no dots," do not also add "no commas" or "no hyphens" as constraints.
- Do NOT say CORRECT if ANY constraint is violated, even a minor one. When in doubt, re-count and re-verify.
- Do NOT unnecessarily correct responses that are already correct. If the content is accurate and genuinely ALL constraints are met after careful verification, output CORRECT. Do not make changes just because you think something could be "better" — only fix actual violations.
- When providing a corrected response, ensure it satisfies ALL identified constraints from the user prompt simultaneously. Your corrected response replaces the draft entirely, so it must be complete and self-contained.
- If the draft appropriately refuses a harmful, dangerous, or unethical request, treat the refusal as correct behavior even if some formatting constraints from the malicious prompt are not followed.
- Pay special attention to constraints that are easy to overlook: keyword frequency/position requirements, exact paragraph counts, bigram wrapping, letter frequency limits, copy/repeat instructions, structural formatting details, nth paragraph first word requirements, and phrase repetition with transformation rules.
- When counting paragraphs, use the separator specified in the prompt (e.g., two new lines). If no separator is specified, use standard paragraph breaks. Be precise about which paragraph is which.
- Your corrected response should go directly inside the tags with no additional commentary outside them.
- Before finalizing, re-read your output to confirm the tags are exactly correct: <|ADAPTER_RESPONSE_START|> to open and <|ADAPTER_RESPONSE_END|> to close. Verify both tags character by character.
"""

dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
                f"User Prompt: {x['messages'][0]['content']}\n<draft_response>{x['claude_response']}</draft_response>\n"
                "/no_think"
            )
        }
    ]
})

# (rohan): dont change these settings!
DEFAULT_MODEL_NAME = "unsloth/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bf16 on H100)
    load_in_4bit=False,  # (rohan): always False!
    fast_inference=True,
    gpu_memory_utilization=0.3,
)

before = len(dataset)
dataset = dataset.filter(
    lambda x: len(tokenizer.apply_chat_template(x["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False)) < DEFAULT_MAX_SEQ_LENGTH
)
print(f"Filtered dataset: {before} -> {len(dataset)} (removed {before - len(dataset)} prompts exceeding {DEFAULT_MAX_SEQ_LENGTH} tokens)")


from api_adapter.ifbench.eval_utils import (
    test_instruction_following_loose,
    InputExample,
    normalize_instruction_kwargs
)

def reward_fn(prompts, completions, ground_truth, key, claude_reward, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    # extract the boxed answer from each response
    rewards = []
    for response, gt, k, p, cr in zip(responses, ground_truth, key, prompts, claude_reward):
        try:
            gt = eval(gt)
            inp = InputExample(
                key=k,
                prompt=p[-1]['content'],
                instruction_id_list=gt[0]['instruction_id'],
                kwargs=normalize_instruction_kwargs(gt[0]['kwargs'])
            )
            adapter_response = ptrn.findall(response)[-1]
            if cr:
                if adapter_response == 'CORRECT': rewards.append(1.0)
                else: rewards.append(0.0)
                continue
            prompt_to_response = {inp.prompt: adapter_response}
            r = test_instruction_following_loose(inp, prompt_to_response)
            rewards.append(float(r.follow_all_instructions))
        except Exception as e:
            # logger.exception(e)
            rewards.append(0.0)
    return rewards


# (rohan): dont change these lora settings!
model = FastLanguageModel.get_peft_model(
    model,
    r=DEFAULT_LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=DEFAULT_LORA_RANK * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


from pathlib import Path
from trl.trainer import GRPOConfig

wandb_run_name = "long_context_test_run"
output_dir = Path(f"outputs/ifbench/{wandb_run_name}")
output_dir.mkdir(parents=True, exist_ok=True)
max_steps = 3000

# (rohan): I want 64 rollouts per prompt. I dont know what hyperparameters to set for this.
# figure it out.

per_device_train_batch_size = 4
gradient_accumulation_steps = 16
learning_rate = 5e-6
num_generations = 64

config = GRPOConfig(
    vllm_enable_sleep_mode=True,
    output_dir=str(output_dir),
    run_name=wandb_run_name,
    max_steps=max_steps,
    use_vllm=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    num_generations=num_generations,
    max_completion_length=2048,
    temperature=1.0,
    top_k=50,
    logging_steps=5,
    save_steps=200,
    bf16=True,
    report_to="wandb",
    log_completions=False,
)
print(config)


from trl.trainer import GRPOTrainer

# Workaround: TRL expects warnings_issued on the model but PEFT doesn't expose it
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
