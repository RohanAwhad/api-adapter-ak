"""
OUTPUT_DIR=outputs/grpo-adapter-only \
WANDB_RUN_NAME=test-v1-no-few-shot-examples-adapter-only \
UNSLOTH_COMPILE_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1 \
python scripts/adapter_only_training.py 2>&1 | tee logs/adapter_only_training.log
"""
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256


import dotenv
dotenv.load_dotenv(override=True)
import os
wandb_project = os.environ["WANDB_PROJECT"]
wandb_entity = os.environ["WANDB_ENTITY"]
wandb_run_name = os.environ["WANDB_RUN_NAME"]

import json
from unsloth import FastLanguageModel

training_data = []
with open('data/train.jsonl', 'r') as f:
    for line in f.readlines():
        if line.strip():
            data = json.loads(line)
            training_data.append(data)

print(training_data[0])


from datasets import Dataset
dataset = Dataset.from_list(training_data)
print(dataset)


SYSTEM_PROMPT = (
    "Evaluate the arithmetic expression using the symbol definitions provided. "
    "There are 4 symbols θ, α, γ, β each representing one of the four basic arithmetic operations (+, -, ×, ÷). "
    "Each symbol maps to exactly one operation. Standard operator precedence (BODMAS) applies. "
    "Be concise. Put your final answer in \\boxed{}."
)

print(SYSTEM_PROMPT)


dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Expression: {x['expression']} ->/no_think"}
    ]
})
print(dataset[0])
dataset.shuffle(seed=23)


from pathlib import Path
from trl.trainer import GRPOConfig

output_dir = Path(os.environ['OUTPUT_DIR'])
max_steps = 1000
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
    per_device_eval_batch_size=per_device_eval_batch_size,
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
    eval_strategy="steps",
    eval_steps=100,
    do_eval=True,
)
print(config)


import re
ptrn = re.compile(r"\\boxed\{(.*)\}")
a = ptrn.search("1 + 2 = \\boxed{3}")
print(a.group(1))
b = ptrn.search("1 + 2 = 3")
print(b)


def correctness_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    # extract the boxed answer from each response
    rewards = []
    for response, ans in zip(responses, answer):
        try:
            _ans = ptrn.search(response)
            rewards.append(float(float(_ans.group(1)) == float(ans)))
        except Exception as e:
            rewards.append(0.0)
        # print(f"response: {response}, true_ans: {ans}, reward: {rewards[-1]}")
    return rewards
    



# test correctness_reward_fn
prompts = ["1 + 2 = \\boxed{3}", "1 + 2 = 3"]
completions = [[{'content':"1 + 2 = \\boxed{3}"}], [{'content':"1 + 2 = 4"}]]
answer = ["3", "4"]
correctness_reward_fn(prompts, completions, answer)


DEFAULT_MODEL_NAME = "unsloth/Qwen3-8B"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bf16 on H100)
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=DEFAULT_LORA_RANK,
    gpu_memory_utilization=0.5,
)

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

FastLanguageModel.for_training(model)


train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(train_dataset, eval_dataset)



from trl.trainer import GRPOTrainer

# Workaround: TRL expects warnings_issued on the model but PEFT doesn't expose it
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_fn],
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
trainer.train()