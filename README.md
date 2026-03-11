# API Adapter

Chain a local LoRA-adapted model to a proprietary API to correct and verify its outputs. A proof-of-concept using synthetic arithmetic with custom symbols.

## The Idea

Large API models (Claude, GPT, etc.) can fail on tasks outside their training distribution. Instead of fine-tuning the API model (which you can't), train a small local model to sit downstream and fix errors. The local adapter sees the API's answer and either accepts it (`CORRECT`) or overrides it with the right answer.

We test this with **custom arithmetic symbols** — a task Claude has never seen:

| Symbol | Operation |
|--------|-----------|
| θ (theta) | addition (+) |
| α (alpha) | subtraction (-) |
| γ (gamma) | multiplication (x) |
| β (beta) | division (/) |

Claude scores **0%** on custom symbol expressions. The adapter learns to correct these while preserving Claude's correct answers on standard arithmetic.

## Architecture

```
Expression ──> Claude API ──> Adapter (Qwen3-8B + LoRA) ──> Final Answer
               (Haiku)         trained via GRPO
```

1. **Claude API** (Haiku via Vertex AI) evaluates the expression — works for standard math, fails on custom symbols
2. **Adapter** (Qwen3-8B, 87M trainable params via LoRA) checks Claude's answer:
   - If correct: outputs `\boxed{CORRECT}` (resolves to Claude's answer)
   - If wrong/missing: computes and outputs `\boxed{answer}`
3. **GRPO training** (TRL + Unsloth + vLLM) with binary correctness reward — no supervised labels needed, just a verifier

## Experiment Design

The adapter prompt tells the model that four symbols each map to one arithmetic operation, but **not which is which**. A few-shot examples give hints:

```
Expression: 3 θ 4 | API answer: 7 → \boxed{CORRECT}     # Claude got it right
Expression: 10 α 3 | API answer: 5 → \boxed{7}           # Claude got it wrong, adapter corrects
Expression: 2 γ 6 | API answer: none → \boxed{12}         # Claude gave no answer, adapter computes
```

Through GRPO, the model must figure out the correct symbol-to-operation mapping purely from the reward signal (1.0 if final answer matches ground truth, 0.0 otherwise). The model is never told the mappings directly — it learns to reason about them from the few-shot examples and the binary reward.

### Ablations

We also tested three other prompt configurations to isolate what matters:

- **Explicit symbols, no CORRECT token**: No learning. Reward flat at ~0.5. Qwen3's thinking mode consumed all tokens before producing an answer.
- **No symbols, no CORRECT token**: Same failure — the model needs some symbol information to even begin.
- **Explicit symbols + CORRECT token**: Reward 1.0 from step 1. The model simply reads the definitions and solves everything — too easy to be interesting.

The vague-symbols setup is the sweet spot: the model has enough information to reason from, but must actually learn *how* to reason through RL.

## Results

### Final Test Accuracy (400 samples: 200 custom + 200 standard)

| | Custom Symbols | Standard Math | Overall |
|---|---|---|---|
| **Claude Haiku (no adapter)** | 0.0% | 75.5% | 37.8% |
| **Adapter @ 500 steps** | 21.5% | 97.5% | 59.5% |
| **Adapter @ 1000 steps** | **84.0%** | **97.5%** | **90.8%** |
| Random baseline (custom) | ~10-15% | — | — |

### Learning Progression

The model learned symbols incrementally through training:

- **Steps 0-500**: Learned γ=x and θ=+ (the two most distinctive mappings). Custom accuracy: 21.5%
- **Steps 500-1000**: Learned α=- and β=÷. Custom accuracy jumped to 84.0%

Custom symbol reward trend across steps 500-1000:

| Training Phase | Avg Custom Reward |
|---|---|
| Steps 500-540 | 0.35 |
| Steps 540-620 | 0.49 |
| Steps 620-720 | 0.40 |
| Steps 720-800 | 0.62 |
| Steps 800-820 | 0.70 |

### Example Outputs (1000 steps)

**Correct predictions:**

```
36 γ 52           => true=1872    claude=None   adapter=1872    (γ=x)
78 α 43 α 58      => true=-23     claude=10     adapter=-23     (α=-)
7 γ 84 θ 43       => true=631     claude=None   adapter=631     (γ=x, θ=+, BODMAS)
50 α 32 γ 33      => true=-1006   claude=None   adapter=-1006   (BODMAS: 50-32x33)
77 γ 77 θ 93 γ 78 => true=13183   claude=77     adapter=13183   (4-term expression)
```

**Failure modes (32/200 = 16%):**

Two patterns account for most errors:

1. **Echoing Claude's wrong answer (~70% of failures)** — the adapter outputs the first number in the expression instead of computing:
   ```
   36 α 85 θ 71  => true=22       adapter=36    (echoed Claude)
   44 α 4        => true=40       adapter=44    (echoed Claude)
   ```

2. **Arithmetic errors (~30% of failures)** — correct symbols, wrong computation on large multi-step expressions:
   ```
   82 γ 46 γ 56 γ 79  => true=16687328  adapter=16707428  (off by 20100)
   47 θ 90 θ 69 α 77  => true=129       adapter=126       (off by 3)
   ```

## Project Structure

```
src/api_adapter/
  symbols.py       # Custom symbol engine (evaluation, precedence, generation)
  api_client.py    # Claude API via Vertex AI (async, batched)
  local_model.py   # Qwen3-8B + LoRA loading, prompt formatting
  reward.py        # Binary correctness reward (handles CORRECT token)
  train.py         # GRPO training pipeline (TRL + Unsloth + vLLM)
  dataset.py       # Synthetic dataset generation
  evaluate.py      # Evaluation utilities

scripts/
  generate_dataset.py    # Generate train/test data (1600/400 samples)
  run_baseline.py        # Run Claude baseline on test set
  train_grpo.py          # CLI for GRPO training
  analyze_condition_d.py # Parse logs, plot rewards, run evaluation
  test_direct.py         # Side-by-side LoRA vs base model comparison

prototype.py    # End-to-end chain test (Claude API -> adapter)
tests/          # 24 unit tests for symbol engine
data/           # Generated datasets (gitignored)
outputs/        # Training outputs, checkpoints, plots (gitignored)
```

## Setup

```bash
# Local development (macOS / no GPU)
pip install -e ".[dev]"
pytest  # runs 24 symbol engine tests

# GPU node (training)
pip install -e ".[gpu,dev]"
```

### Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_REGION="your-gcp-region"
```

### Training

```bash
# Generate dataset (requires Claude API access via Vertex AI)
python scripts/generate_dataset.py
python scripts/run_baseline.py

# Train (vague symbols + CORRECT token)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --condition D --max-steps 1000

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
    --condition D --max-steps 1000 \
    --resume outputs/grpo_condition_D/checkpoint-500

# Evaluate
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_condition_d.py
```

## Technical Details

### Model & Training

- **Base model**: Qwen3-8B (`unsloth/Qwen3-8B`) — NOT Qwen3.5-9B (vision model, crashes)
- **LoRA**: rank 32, alpha 64, targeting all attention + MLP projections (87M trainable / 8.3B total = 1.05%)
- **Training**: GRPO via TRL 0.29, Unsloth 2026.3.4, vLLM 0.17 for fast generation
- **Hardware**: Single NVIDIA H100 80GB
- **Batch size**: 16, with 64 generations per prompt for GRPO
- **Optimizer**: AdamW 8-bit, lr=5e-6, linear warmup 10%

### Key Configuration

- `chat_template_kwargs={"enable_thinking": False}` — required for Qwen3, otherwise thinking mode consumes all tokens
- `gpu_memory_utilization=0.3` — vLLM KV cache fraction; 0.6 causes OOM when sharing GPU with training
- `TORCHDYNAMO_CACHE_SIZE_LIMIT=256` — prevents `FailOnRecompileLimitHit` crash from varying completion lengths during long runs
- `save_steps=200` — checkpoint frequency for crash recovery
