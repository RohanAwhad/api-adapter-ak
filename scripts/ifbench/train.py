import sys
from loguru import logger

logger.remove()

# setup a file handler
file_handler = logger.add("logs/ifbench/train.log")


from datasets import Dataset

dataset = Dataset.from_json('data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl')


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



import torch._dynamo
torch._dynamo.config.cache_size_limit = 256


import dotenv
dotenv.load_dotenv(override=True)

from unsloth import FastLanguageModel



import re
ptrn = re.compile(r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL)



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


DEFAULT_MODEL_NAME = "unsloth/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bf16 on H100)
    load_in_4bit=False,
    gpu_memory_utilization=0.5,
)


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
            logger.exception(e)
            rewards.append(0.0)
    return rewards

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

output_dir = Path("outputs/ifbench/test-run")
output_dir.mkdir(parents=True, exist_ok=True)
wandb_run_name = "test-run"
max_steps = 3000
per_device_train_batch_size = 16
per_device_eval_batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 5e-6
num_generations = 64

config = GRPOConfig(
    output_dir=str(output_dir),
    run_name=wandb_run_name,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    # per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    num_generations=num_generations,
    generation_batch_size=num_generations,
    max_completion_length=512,
    temperature=1.0,
    top_k=50,
    logging_steps=5,
    save_steps=200,
    bf16=True,
    report_to="wandb",
    log_completions=True,
    num_completions_to_print=5,
    # eval_strategy="steps",
    # eval_steps=100,
    # do_eval=True,
)


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

model.save_pretrained(output_dir / "final_adapter")
tokenizer.save_pretrained(output_dir / "final_adapter")
