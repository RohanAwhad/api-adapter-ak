"""
OUTPUT_DIR=outputs/grpo-api-adapter-post-training \
WANDB_RUN_NAME=test-v2-no-few-shot-examples-api-adapter-post-training \
CUDA_VISIBLE_DEVICES=0 \
python scripts/api_adapter_training.py 2>&1 | tee logs/api_adapter_training.log
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

import re
ptrn = re.compile(r"\\boxed\{(.*)\}")

def generate_training_data():
    training_data = []
    with open('data/train.jsonl', 'r') as f:
        for line in f.readlines():
            if line.strip():
                data = json.loads(line)
                training_data.append(data)


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


    # Build API prompt

    dataset = dataset.map(lambda x: {
        "claude_messages": [
            {"role": "user", "content": f"Write down your solution and steps taken to reach the solution. You have hard context limit of 500 tokens on top of the 10000 tokens for thinking.\nExpression: {x['expression']} ->"}
        ]
    })
    dataset = dataset.map(lambda x: {'claude_system_prompt': SYSTEM_PROMPT})
    print(dataset[0]['claude_messages'][0]['content'])


    os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
    os.environ["GOOGLE_CLOUD_REGION"] = os.environ["CLOUD_ML_REGION"]


    # Generate api completions
    import asyncio
    from anthropic import AsyncAnthropicVertex

    client = AsyncAnthropicVertex(
        region=os.environ["GOOGLE_CLOUD_REGION"],
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
    )

    # simple completion
    async def test_completion():
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{'role': 'user', 'content': 'say: Hello, world!'}],
            system="You are a helpful assistant.",
        )
        return response.content[0].text

    print(asyncio.run(test_completion()))


    # reasoning completion
    async def test_reasoning_completion():
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=11000,
            messages=dataset[0]['claude_messages'],
            system=dataset[0]['claude_system_prompt'],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        return response

    print(asyncio.run(test_reasoning_completion()))


    # generate completions
    semaphore = asyncio.Semaphore(40)

    async def generate_completions_single(x):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=11000,
                    messages=x['claude_messages'],
                    system=x['claude_system_prompt'],
                    thinking={"type": "enabled", "budget_tokens": 10000},
                )
                return response.content[-1].text
        except Exception as e:
            return f"Error generating completion for {x['expression']}: {e}"

    async def generate_all_completions():
        return await asyncio.gather(*[generate_completions_single(x) for x in dataset])

    completions = asyncio.run(generate_all_completions())
    print(completions[0])


    dataset = dataset.add_column('claude_response', completions)
    print(dataset[0])


    print(dataset)




    def extract_answer(response):
        try:
            return ptrn.search(response).group(1).replace(',', '')
        except:
            return 'null'

    dataset = dataset.map(lambda x: {'claude_answer': extract_answer(x['claude_response'])})
    print(dataset[0])


    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Expression: {x['expression']}\n<draft_response>{x['claude_answer']}</draft_response>\nLook at the expression, and draft response and output \\boxed{{CORRECT}} if the final answer is correct or if its incorrect, output the corrent final answer \\boxed{{n}}, where n is the corrent final answer."}
        ]
    })
    print(dataset[0])


    dataset.shuffle(seed=23)


    # save dataset
    dataset.to_json("data/train_api_adapter.jsonl")


# generate_training_data()

# load the dataset
from datasets import Dataset
dataset = Dataset.from_json("data/train_api_adapter.jsonl")
print(dataset[0])


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


def correctness_reward_fn_strict(prompts, completions, answer, claude_answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    # extract the boxed answer from each response
    rewards = []
    for response, ans, cans in zip(responses, answer, claude_answer):
        try:
            _ans = ptrn.findall(response)

            try:
                is_claude_correct = float(cans) == float(ans)
            except:
                is_claude_correct = False

            adapter_ans = _ans[-1].replace(',', '')

            if is_claude_correct and adapter_ans == 'CORRECT': rewards.append(1.0)
            elif not is_claude_correct and float(adapter_ans) == float(ans): rewards.append(1.0)
            else: rewards.append(0.0)
        except Exception as e:
            rewards.append(0.0)
    return rewards



# test correctness_reward_fn
prompts = ["1 + 2 = \\boxed{3}", "1 + 2 = 3"]
adapter_completions = [
    [{'content':"\\boxed{3}"}],
    [{'content':"\\boxed{4}"}], [{'content':"\\boxed{5,390}"}], [{'content':"\\boxed{5}"}], [{'content':"\\boxed{CORRECT}"}], [{'content':"\\boxed{6}"}],
    [{'content':"The final answer is $\\boxed{-44}$.\n\nSince the draft response is correct, the output is:\n\n$$\n\\boxed{CORRECT}\n$$"}],
]
answer = [3, 4, 5390, 5, 6, 6, -44]
claude_answer = ['3', '3', '5390', '6', '6', 'null', '-44']
print(correctness_reward_fn_strict(prompts, adapter_completions, answer, claude_answer))


DEFAULT_MODEL_NAME = "outputs/grpo-adapter-only/checkpoint-1000/"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bf16 on H100)
    load_in_4bit=False,
    gpu_memory_utilization=0.5,
)

# because the model was mid-trained using peft, we don't need to add the lora layers again
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=DEFAULT_LORA_RANK,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha=DEFAULT_LORA_RANK * 2,
#     lora_dropout=0,
#     bias="none",
#     use_gradient_checkpointing="unsloth",
#     random_state=3407,
# )

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
    reward_funcs=[correctness_reward_fn_strict],
    args=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)
trainer.train()
