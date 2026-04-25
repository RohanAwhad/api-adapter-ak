---
name: ifbench-eval-openai
description: Run IFBench evaluation using an OpenAI-compatible API as the adapter model. Use when the user wants to evaluate a model checkpoint on IFBench via an OpenAI-compatible endpoint (e.g., vLLM).
argument-hint: [model-name] [base-url]
arguments: [model_name, base_url]
---

# IFBench OpenAI Evaluation

Run `scripts/ifbench/eval_openai.py` to evaluate an adapter model served via an OpenAI-compatible API on the IFBench benchmark.

## What the user provides (ask if missing)

1. **Model name** (`$model_name`): The model name served by the API (e.g., `qwen3-4b-step1100`)
2. **Base URL** (`$base_url`): The OpenAI-compatible server URL (e.g., `http://10.241.128.25:8000/v1`)

## Optional parameters (use defaults if not specified)

- **Tokenizer**: HuggingFace tokenizer name (default: `Qwen/Qwen3-4B`)
- **API key**: OpenAI API key (default: `dummy` for local vLLM)
- **Claude response path**: Path to claude response jsonl (default: `data/ifbench/api.jsonl`)
- **Max sequence length**: Max seq length for filtering long prompts (default: `4096`)
- **Concurrency**: Number of concurrent API requests (default: `40`)

## How to run

Ask the user if they want to run it in the current shell or in a tmux session. Then execute:

```bash
OPENAI_BASE_URL=$base_url \
OPENAI_API_KEY=dummy \
MODEL_NAME=$model_name \
TOKENIZER_NAME=Qwen/Qwen3-4B \
python scripts/ifbench/eval_openai.py
```

Override optional params by adding env vars: `CLAUDE_RESPONSE_PATH`, `MAX_SEQ_LENGTH`, `CONCURRENCY`.

## Output

The script will:
1. Load IFBench test data (300 prompts) and Claude draft responses
2. Build adapter prompts with the GEPA-optimized system prompt
3. Generate completions via the OpenAI API
4. Post-process adapter responses (CORRECT / corrected)
5. Run the IFBench evaluator (loose + strict)
6. Print prompt-level and instruction-level accuracy

Results are saved to `data/ifbench/api_adapter_v10_<model_name>.jsonl` and evaluation results to `data/ifbench/api_adapter_v10_<model_name>_evaluation/`.
