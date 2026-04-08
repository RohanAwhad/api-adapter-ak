import os
import re
import sys
from pathlib import Path

import dotenv
import torch._dynamo
from datasets import Dataset
from loguru import logger
from trl.trainer import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from api_adapter.ifbench.eval_utils import (
    test_instruction_following_loose,
    InputExample,
    normalize_instruction_kwargs
)

# ===
# Configuration
# ===

LOG_FILE = "logs/ifbench/train_script.log"
CUDA_DEVICES = '1'

DYNAMO_CACHE_SIZE_LIMIT = 256
DATASET_PATH = 'data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl'

MODEL_NAME = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = False
GPU_MEMORY_UTILIZATION = 0.5

LORA_RANK = 32
LORA_ALPHA = LORA_RANK * 2

WANDB_RUN_NAME = "test-run-v2"
OUTPUT_DIR = Path(f"outputs/ifbench/{WANDB_RUN_NAME}")
MAX_STEPS = 3000
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-6
NUM_GENERATIONS = 64
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "linear"
OPTIM = "adamw_8bit"
MAX_COMPLETION_LENGTH = 512
TEMPERATURE = 1.0
TOP_K = 50
LOGGING_STEPS = 5
SAVE_STEPS = 200
BF16 = True
REPORT_TO = "wandb"
NUM_COMPLETIONS_TO_PRINT = 5

SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to look at the user prompt and the draft response and output <|ADAPTER_RESPONSE_START|>CORRECT<|ADAPTER_RESPONSE_END|> if the draft response is correct
If the draft response is incorrect, output the correct final answer <|ADAPTER_RESPONSE_START|>final_answer<|ADAPTER_RESPONSE_END|>, where final_answer is the corrent final answer.

Example:
User Prompt: What is the capital of France?
Draft Response: The capital of France is Paris.
Output: <|ADAPTER_RESPONSE_START|>CORRECT<|ADAPTER_RESPONSE_END|>

User Prompt: What is the capital of France?
Draft Response: The capital of France is London.
Output: The capital of France is Paris but the draft response says its London. So it is incorrect. <|ADAPTER_RESPONSE_START|>The capital of France is Paris.<|ADAPTER_RESPONSE_END|>
""".strip()

# ===

logger.remove()
file_handler = logger.add(LOG_FILE)

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICES

torch._dynamo.config.cache_size_limit = DYNAMO_CACHE_SIZE_LIMIT

dotenv.load_dotenv(override=True)

dataset = Dataset.from_json(DATASET_PATH)

ptrn = re.compile(r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL)

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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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
            logger.exception(e)
            rewards.append(0.0)
    return rewards

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    run_name=WANDB_RUN_NAME,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    # per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    optim=OPTIM,
    num_generations=NUM_GENERATIONS,
    generation_batch_size=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    bf16=BF16,
    report_to=REPORT_TO,
    log_completions=True,
    num_completions_to_print=NUM_COMPLETIONS_TO_PRINT,
    # eval_strategy="steps",
    # eval_steps=100,
    # do_eval=True,
)


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

model.save_pretrained(OUTPUT_DIR / "final_adapter")
tokenizer.save_pretrained(OUTPUT_DIR / "final_adapter")
