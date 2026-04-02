from datasets import Dataset
from pathlib import Path

output_dir = Path('data/cuad/reformatted')
test_dataset = Dataset.from_json(str(output_dir / 'test.json'))
print(test_dataset[0])

import os
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
os.environ["GOOGLE_CLOUD_REGION"] = os.environ["CLOUD_ML_REGION"]

# Generate api completions
from anthropic import AsyncAnthropicVertex

client = AsyncAnthropicVertex(
    region=os.environ["GOOGLE_CLOUD_REGION"],
    project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
)

user_prompt_template = """# Context

{context}

---

Question: {question}

---

* Respond with the final answer inside the <final_answer> tag.
* Respond in bullet points.
* The answer should be as-is from the context.

Example:
<final_answer>
- answer 1
- answer 2
- answer 3
</final_answer>
"""


import re
pattern = re.compile(r'<final_answer>(.*?)</final_answer>', re.DOTALL)

def extract_answer(completion):
    matches = pattern.findall(completion)
    answer = matches[-1]
    answer = answer.strip().split('\n')
    answer = [a.strip() for a in answer if a.strip()]
    answer = [a[2:] for a in answer]
    return answer


# evaluate
def evaluate_answer(prompts, completions, answers, **kwargs):
    """
    Evaluate the answer against the reference.
    """
    rewards = []
    for c, a in zip(completions, answers):
        try:
            c_answer = extract_answer(c[0]['content'])
            a_idx_to_c_idx = {}
            c_indices = set()
            for part in a:
                for c_idx, c_part in enumerate(c_answer):
                    if part in c_part and c_idx not in c_indices:
                        a_idx_to_c_idx[part] = c_idx
                        c_indices.add(c_idx)

            # if all parts are matched, and no extra parts are predicted
            if len(a_idx_to_c_idx) == len(a) and len(c_indices) == len(c_answer): rewards.append(1.0)
            else: rewards.append(0.0)
        except:
            rewards.append(0.0)

    return rewards

test_dataset = test_dataset.map(lambda x: {'claude_messages': [{'role': 'user', 'content': user_prompt_template.format(context=x['context'], question=x['question'])}]})

# generate completions
import asyncio
from tqdm import tqdm

semaphore = asyncio.Semaphore(40)
pbar = tqdm(desc='Generating completions', total=len(test_dataset))

async def generate_completions(x):
    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=11000,
                messages=x['claude_messages'],
                thinking={"type": "enabled", "budget_tokens": 10000}, 
            )
            pbar.update(1)
            return response.content[-1].text
        except Exception as e:
            pbar.update(1)
            return f"Error generating completion for {x['question']}: {e}"

async def generate_all_completions():
    return await asyncio.gather(*[generate_completions(x) for x in test_dataset])

completions = asyncio.run(generate_all_completions())
test_dataset = test_dataset.add_column('claude_completion', completions)
pbar.close()

# save
test_dataset.to_json(str(output_dir / 'test_with_claude_completions.json'))

# evaluate

tmp = test_dataset[:]
rewards = evaluate_answer(tmp['claude_messages'], tmp['claude_completion'], tmp['answer'])
print(f"Accuracy: {sum(rewards) / len(rewards):.2%}")