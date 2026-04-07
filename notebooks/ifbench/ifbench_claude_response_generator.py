from datasets import Dataset

dataset = Dataset.from_json("data/ifbench/input_train_data.jsonl")

import os
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
os.environ["GOOGLE_CLOUD_REGION"] = os.environ["CLOUD_ML_REGION"]
# Generate api completions
from anthropic import AsyncAnthropicVertex

client = AsyncAnthropicVertex(
    region=os.environ["GOOGLE_CLOUD_REGION"],
    project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
)

async def get_claude_response(messages):
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=11000,
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )
    return response.content[-1].text



# lets do it for a small subset

import asyncio
from tqdm import tqdm

dataset = dataset.select(range(5000))
semaphore = asyncio.Semaphore(40)
pbar = tqdm(total=len(dataset), desc="Processing", ncols=80)

async def get_claude_response_with_semaphore(messages):
    async with semaphore:
        try:
            ret = await get_claude_response(messages)
            return ret
        except Exception:
            return 'Error'
        finally:
            pbar.update(1)

async def async_gather_wrapper():
    return await asyncio.gather(*[get_claude_response_with_semaphore(x['messages']) for x in dataset])

claude_responses = asyncio.run(async_gather_wrapper())
pbar.close()

print(claude_responses[0])

import sys
sys.path.append("notebooks/ifbench/src")
from eval_utils import convert_to_input_example, test_instruction_following_loose

dataset = dataset.add_column('claude_response', claude_responses)
claude_rewards = []
for x in dataset:
    inp = convert_to_input_example(x)
    prompt_to_response = {inp.prompt: x['claude_response']}
    r = test_instruction_following_loose(inp, prompt_to_response)
    claude_rewards.append(r.follow_all_instructions)

dataset = dataset.add_column('claude_reward', claude_rewards)
print(dataset[0])

dataset.to_json("data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl")
