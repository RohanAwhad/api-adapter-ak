import argparse
import asyncio
import dataclasses
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

from api_adapter.ifbench.eval_utils import (
    InputExample,
    normalize_instruction_kwargs,
    test_instruction_following_loose,
)


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
- Check paragraph separators match requirements (e.g., markdown divider `***` or `\\n\\n`)
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
""".strip()

MODEL_NAME = "qwen3-4b"
GROUP_SIZE = 64

_ADAPTER_RESPONSE_PATTERN = re.compile(
    r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL
)


def extract_adapter_response(response: str) -> str:
    try:
        return _ADAPTER_RESPONSE_PATTERN.findall(response)[-1]
    except:
        return ""


@dataclasses.dataclass
class AssistantMessage:
    role: str
    content: str


def reward_fn(completion: list[AssistantMessage], state: dict, info: dict) -> float:
    try:
        response = extract_adapter_response(completion[-1].content) or ""
        if response is None or not response.strip():
            return 0.0

        gt = eval(info['ground_truth'])
        input_example = InputExample(
            key=info["key"],
            instruction_id_list=gt[0]['instruction_id'],
            prompt=info['messages'][-1]['content'],
            kwargs=normalize_instruction_kwargs(gt[0]['kwargs']),
        )

        state['lgtm_count'] = 1 if response.strip() == 'CORRECT' else 0
        state['fixme_count'] = 1 if response.strip() != 'CORRECT' else 0

        if info['claude_reward']:
            if response.strip() == "CORRECT":
                return 1.0
            else:
                return 0.0

        if response.strip() == "CORRECT":
            return 0.0
        if response.strip() == info['claude_response']:
            return 0.0

        reward = 0.5
        prompt_to_response = {input_example.prompt: response}
        output_example = test_instruction_following_loose(input_example, prompt_to_response)
        return reward + (float(output_example.follow_all_instructions) / 2)
    except:
        return 0.0


def load_and_prepare_data(num_samples: int) -> Dataset:
    dataset = Dataset.from_json('data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl')
    dataset = dataset.map(lambda x: {
        "question": (
            f"User Prompt: {x['messages'][0]['content']}\n"
            f"<draft_response>{x['claude_response']}</draft_response>\n"
            "/think"
        ),
        "answer": "",
        "info": {**x},
    })

    lgtm_dataset = dataset.filter(lambda x: x['claude_reward'] == True)
    fixme_dataset = dataset.filter(lambda x: x['claude_reward'] == False)

    lgtm_train, _ = lgtm_dataset.train_test_split(test_size=0.2, seed=42).values()
    fixme_train, _ = fixme_dataset.train_test_split(test_size=0.2, seed=42).values()

    train_dataset = concatenate_datasets([lgtm_train, fixme_train])
    subset = train_dataset.shuffle(seed=42).select(range(num_samples))
    return subset


async def generate_rollouts(subset: Dataset) -> list[list[str]]:
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="")
    semaphore = asyncio.Semaphore(40)

    async def generate_rollout(prompt: str, pbar=None) -> str:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=16384,
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.exception(f"Error in generate_rollout: {e}")
            return ""
        finally:
            if pbar:
                pbar.update(1)

    async def generate_group(prompt: str, pbar=None) -> list[str]:
        tasks = [generate_rollout(prompt, pbar) for _ in range(GROUP_SIZE)]
        return await asyncio.gather(*tasks)

    total = len(subset) * GROUP_SIZE
    pbar = tqdm(total=total, desc="Generating groups", ncols=50)
    tasks = [generate_group(x['question'], pbar) for x in subset]
    groups = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()
    return groups


def calculate_rewards(dataset: Dataset) -> list[list[float]]:
    group_rewards = []
    for x in dataset:
        info = {**x}
        rewards = []
        for s in x['groups']:
            state = {}
            reward = reward_fn(
                completion=[AssistantMessage(role="assistant", content=s)],
                state=state,
                info=info,
            )
            rewards.append(reward)
        group_rewards.append(rewards)
    return group_rewards


def calculate_pass_at_n(dataset: Dataset) -> tuple[Dataset, dict]:
    N = [1, 2, 8, 16, 32, 64]

    group_pass_at_n = []
    for x in dataset:
        pass_at_n = [0] * len(N)
        for i, r in enumerate(x['group_rewards']):
            if r == 1.0:
                for j, n in enumerate(N):
                    if i <= n:
                        pass_at_n[j] = 1
        group_pass_at_n.append(pass_at_n)

    dataset = dataset.add_column('group_pass_at_n', group_pass_at_n)

    results = {}
    for i, n in enumerate(N):
        values = [x['group_pass_at_n'][i] for x in dataset]
        mean_val = float(np.mean(values))
        results[f"pass@{n}"] = mean_val
        logger.info(f"Pass @ {n}: {mean_val}")

    return dataset, results


def calculate_advantages(dataset: Dataset) -> Dataset:
    advantages = []
    for x in dataset:
        advantage = np.array(x['group_rewards']) - np.mean(x['group_rewards'])
        advantages.append(advantage.tolist())
    dataset = dataset.add_column('advantages', advantages)
    return dataset


def plot_advantages(dataset: Dataset, output_path: str) -> None:
    all_advantages = []
    for x in dataset:
        all_advantages.extend(x['advantages'])
    all_advantages = np.array(all_advantages)

    plt.figure(figsize=(10, 6))
    plt.hist(all_advantages, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Advantage')
    plt.ylabel('Density')
    plt.title('Density Plot of Advantages')
    plt.ylim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="IFBench EDA: qwen3-4b thinking mode evaluation")
    parser.add_argument('--num-samples', type=int, default=500)
    args = parser.parse_args()

    results_dir = 'results/ifbench/qwen3_4b_thinking'
    os.makedirs(results_dir, exist_ok=True)

    data_path = f'data/ifbench/subset_train_dataset_with_its_groups_gepa_prompt_think_{MODEL_NAME}.jsonl'

    logger.info(f"Starting pipeline with {args.num_samples} samples, GROUP_SIZE={GROUP_SIZE}")

    subset = load_and_prepare_data(args.num_samples)
    logger.info(f"Loaded {len(subset)} samples")

    groups = await generate_rollouts(subset)
    subset = subset.add_column('groups', groups)

    subset.to_json(data_path, lines=True)
    logger.info(f"Rollouts saved to {data_path}")

    group_rewards = calculate_rewards(subset)
    subset = subset.add_column('group_rewards', group_rewards)

    subset, pass_at_n_results = calculate_pass_at_n(subset)

    metrics_path = f'{results_dir}/pass_rate.txt'
    with open(metrics_path, 'w') as f:
        for k, v in pass_at_n_results.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Pass rate saved to {metrics_path}")

    subset = calculate_advantages(subset)

    plot_path = f'{results_dir}/advantages_density.png'
    plot_advantages(subset, plot_path)

    subset.to_json(data_path, lines=True)
    logger.info(f"Final data saved to {data_path}")

    logger.info("Pipeline complete!")


if __name__ == '__main__':
    asyncio.run(main())
