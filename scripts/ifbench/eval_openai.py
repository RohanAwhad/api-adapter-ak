"""
Evaluate IFBench using an OpenAI-compatible API as the adapter model.

Env vars:
  OPENAI_BASE_URL    – base URL of the OpenAI-compatible server
  OPENAI_API_KEY     – API key (use 'dummy' for local vLLM)
  MODEL_NAME         – model name served by the API
  TOKENIZER_NAME     – HuggingFace tokenizer name (for token counting / chat template)
  CLAUDE_RESPONSE_PATH – path to claude response jsonl (default: data/ifbench/api.jsonl)
  MAX_SEQ_LENGTH     – max sequence length for filtering (default: 4096)
  CONCURRENCY        – number of concurrent requests (default: 40)

Example:
  OPENAI_BASE_URL=http://10.241.128.25:8000/v1 \
  OPENAI_API_KEY=dummy \
  MODEL_NAME=qwen3-4b-step1100 \
  TOKENIZER_NAME=Qwen/Qwen3-4B \
  python scripts/ifbench/eval_openai.py
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

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

ADAPTER_RESPONSE_PATTERN = re.compile(
    r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL
)


def build_adapter_prompt(user_prompt: str, draft_response: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User Prompt: {user_prompt}\n"
                f"<draft_response>{draft_response}</draft_response>"
                "/no_think"
            ),
        },
    ]


def load_claude_responses(path: str) -> dict[str, str]:
    mapping = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            mapping[row["prompt"]] = row["response"]
    return mapping


async def main():
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    model_name = os.environ["MODEL_NAME"]
    tokenizer_name = os.environ["TOKENIZER_NAME"]
    claude_response_path = os.environ.get(
        "CLAUDE_RESPONSE_PATH", "data/ifbench/api.jsonl"
    )
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
    concurrency = int(os.environ.get("CONCURRENCY", "40"))

    print(f"model_name: {model_name}")
    print(f"tokenizer_name: {tokenizer_name}")
    print(f"base_url: {base_url}")
    print(f"claude_response_path: {claude_response_path}")
    print(f"max_seq_length: {max_seq_length}")
    print(f"concurrency: {concurrency}")

    # --- load data ---
    dataset = load_dataset("allenai/IFBench_test", split="train")
    claude_responses = load_claude_responses(claude_response_path)

    mapped_claude_responses = []
    for x in dataset:
        mapped_claude_responses.append(claude_responses[x["prompt"]])
    dataset = dataset.add_column("claude_response", mapped_claude_responses)

    # --- build adapter prompts ---
    dataset = dataset.map(
        lambda x: {
            "adapter_prompt": build_adapter_prompt(x["prompt"], x["claude_response"])
        }
    )

    # --- tokenizer for filtering + token counting ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    items_to_generate = []
    skipped_items = []
    for x in dataset:
        text = tokenizer.apply_chat_template(
            x["adapter_prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tok_len = len(tokenizer(text)["input_ids"])
        if tok_len >= max_seq_length:
            skipped_items.append(
                {"prompt": x["prompt"], "response": x["claude_response"]}
            )
        else:
            items_to_generate.append(x)

    print(f"items to generate: {len(items_to_generate)}, skipped: {len(skipped_items)}")

    # --- generate completions ---
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(
        desc="Generating completions", total=len(items_to_generate), ncols=80
    )

    async def generate_completion(x):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=x["adapter_prompt"],
                    max_tokens=1024,
                    temperature=1.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating completion for {x['prompt']}: {e}"
            finally:
                pbar.update(1)

    completions = await asyncio.gather(
        *[generate_completion(x) for x in items_to_generate],
        return_exceptions=True,
    )
    pbar.close()

    # --- post-process adapter responses ---
    didnt_find_tags = 0
    lgtm_count = 0
    attempted_correction = 0
    token_cnts = []
    final_outputs = list(skipped_items)

    for completion, x in zip(completions, items_to_generate):
        if isinstance(completion, Exception):
            completion = str(completion)

        token_cnts.append(len(tokenizer(completion)["input_ids"]))
        try:
            adapter_response = ADAPTER_RESPONSE_PATTERN.findall(completion)[-1].strip()
            if adapter_response == "CORRECT":
                final_outputs.append(
                    {"prompt": x["prompt"], "response": x["claude_response"]}
                )
                lgtm_count += 1
            else:
                final_outputs.append(
                    {"prompt": x["prompt"], "response": adapter_response}
                )
                attempted_correction += 1
        except IndexError:
            didnt_find_tags += 1
            final_outputs.append(
                {"prompt": x["prompt"], "response": x["claude_response"]}
            )

    token_cnts = np.array(token_cnts)
    print(f"didnt_find_tags: {didnt_find_tags}")
    print(f"lgtm_count: {lgtm_count}")
    print(f"attempted_correction: {attempted_correction}")
    print(f"median_token_cnt: {np.median(token_cnts)}")
    print(f"avg_token_cnt: {np.mean(token_cnts)}")

    # --- save responses ---
    outdir = "data/ifbench/"
    os.makedirs(outdir, exist_ok=True)
    output_name = f"api_adapter_v10_{model_name}"
    generated_data_path = os.path.join(outdir, f"{output_name}.jsonl")
    with open(generated_data_path, "w") as f:
        f.write("\n".join(json.dumps(row) for row in final_outputs))
    print(f"saved responses to {generated_data_path}")

    # --- run IFBench evaluation ---
    eval_dir = os.path.join(outdir, f"{output_name}_evaluation/")
    os.makedirs(eval_dir, exist_ok=True)
    input_data_path = os.path.join(outdir, "input_test_data.jsonl")

    eval_cmd = [
        "external_repos/IFBench/.venv/bin/python",
        "-m",
        "run_eval",
        f"--input_data={input_data_path}",
        f"--input_response_data={generated_data_path}",
        f"--output_dir={eval_dir}",
    ]
    print(f"running: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)

    # --- print results ---
    for mode in ("loose", "strict"):
        result_path = Path(eval_dir) / f"{output_name}-eval_results_{mode}.jsonl"
        rows = [
            json.loads(line)
            for line in result_path.read_text().splitlines()
            if line.strip()
        ]
        prompt_level = sum(r["follow_all_instructions"] for r in rows) / len(rows)
        instruction_level = sum(
            sum(r["follow_instruction_list"]) for r in rows
        ) / sum(len(r["follow_instruction_list"]) for r in rows)
        print(f"prompt-level {mode}: {prompt_level * 100:.2f}%")
        print(f"instruction-level {mode}: {instruction_level * 100:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
