---
name: gepa-optimize
description: Generate a GEPA prompt optimization script. Use when the user wants to optimize a system prompt using the GEPA library with a task LM, reflection LM, evaluator, and train/val data.
allowed-tools: Bash(python3:*) Read Grep Write Edit
argument-hint: [description-of-optimization-task]
---

# GEPA Prompt Optimization Script Generator

You are generating a complete Python script that runs GEPA prompt optimization. GEPA iteratively improves a system prompt by having a reflection LM analyze failures from a task LM and propose better prompts.

## What the user provides (ask if missing)

The user describes what they want to optimize. You need to determine or ask for:

1. **Task description**: What the task LM should do (e.g., "verify/correct draft responses", "answer math questions")
2. **Seed system prompt**: The initial prompt to optimize (a string)
3. **Task LM**: How to run the model being prompted. One of:
   - A local model via vLLM/Unsloth (specify model name, GPU, max_seq_length)
   - An API model via Anthropic/OpenAI/litellm (specify model name)
4. **Reflection LM**: The stronger model that rewrites prompts (typically Claude Opus via Anthropic API)
5. **Data**: Path to a dataset file (JSONL, JSON, CSV) and how to extract `input` and `answer` fields
6. **Evaluator**: How to score a response — either:
   - A custom evaluation function the user describes
   - An existing function in the codebase (ask for import path)
7. **WandB config** (optional): project name, entity, run name

## Script structure

Generate a single Python script with these sections, in order:

### 1. Imports and setup

```python
import os
import re
import gepa
from gepa.adapters.default_adapter.default_adapter import EvaluationResult
```

### 2. Data loading

Load the dataset and split into train/val. The data must be shaped as:

```python
# Each item: {"input": str, "answer": <any — passed to evaluator>}
trainset: list[dict]  # list of {"input": ..., "answer": ...}
valset: list[dict]     # list of {"input": ..., "answer": ...}
```

The `input` field is the user message content (without system prompt — GEPA prepends the system prompt).
The `answer` field can be any structure the evaluator needs for scoring.

### 3. Seed prompt

```python
seed_prompt = {"system_prompt": "<the initial system prompt>"}
```

The dict key must match what the reflection LM will optimize. For single-prompt optimization, use `"system_prompt"`.

### 4. Task LM callable

Must match the protocol:
```python
def task_lm_callable(messages: list[dict[str, str]]) -> str:
    """
    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
    Returns:
        The model's response as a string
    """
```

**For local vLLM/Unsloth models:**
```python
from unsloth import FastLanguageModel
from vllm import SamplingParams

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="<model_name>",
    max_seq_length=<seq_len>,
    dtype=None,
    load_in_4bit=False,
    gpu_memory_utilization=0.5,
    fast_inference=True,
)
FastLanguageModel.for_inference(model)

def task_lm_callable(messages) -> str:
    texts = [tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=8192)
    outputs = model.fast_generate(texts, sampling_params=sampling_params)
    return outputs[0].outputs[0].text
```

**For API models (Anthropic):**
```python
from anthropic import AnthropicVertex  # or Anthropic

def task_lm_callable(messages) -> str:
    system_prompt = None
    final_messages = []
    for m in messages:
        if m['role'] == 'system':
            if system_prompt is None: system_prompt = m['content']
        else:
            final_messages.append(m)

    kwargs = {"model": "<model>", "max_tokens": 8192, "messages": final_messages}
    if system_prompt: kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[-1].text
```

### 5. Reflection LM callable

Must match the protocol:
```python
def reflection_lm_callable(prompt: str | list[dict[str, str]]) -> str:
    """
    Args:
        prompt: Either a plain string or a list of chat messages
    Returns:
        The reflection model's response as a string
    """
```

Typically uses a strong model (Claude Opus) with extended thinking:
```python
def reflection_lm_callable(prompt):
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    system_prompt = None
    final_messages = []
    for m in prompt:
        if m['role'] == 'system':
            if system_prompt is None: system_prompt = m['content']
        else:
            final_messages.append(m)

    kwargs = {
        "model": "claude-opus-4-6",
        "max_tokens": 64000,
        "messages": final_messages,
        "thinking": {"type": "adaptive"},
    }
    if system_prompt: kwargs["system"] = system_prompt
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    return response.content[-1].text
```

### 6. Reflection prompt template (optional)

A string template with two placeholders that GEPA fills in:
- `<curr_param>` — the current system prompt being optimized
- `<side_info>` — minibatch of inputs, outputs, and feedback

If the user doesn't provide one, use this default:

```python
reflection_prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""
```

### 7. Evaluator callable

Must match the protocol:
```python
def evaluate_callable(data: dict, response: str) -> EvaluationResult:
    """
    Args:
        data: A single item from trainset/valset (has 'input' and 'answer' keys)
        response: The task LM's raw string output
    Returns:
        EvaluationResult(score=float, feedback=str, objective_scores=dict|None)
    
    - score: 0.0 to 1.0 (1.0 = perfect)
    - feedback: Human-readable string explaining what went wrong/right.
              This is what the reflection LM sees to improve the prompt.
              Make it specific and actionable (e.g., "Did not follow instruction: word_count"
              not just "Wrong").
    - objective_scores: Optional dict for multi-objective optimization
    """
```

The feedback quality is critical — it's the primary signal the reflection LM uses to improve the prompt. Good feedback names specific failure modes.

### 8. WandB setup (optional)

```python
import dotenv
dotenv.load_dotenv(override=True)

wandb_kwargs = {'project': '<project>', 'entity': '<entity>'}
os.environ['WANDB_RUN_NAME'] = '<run_name>'
```

### 9. Run optimization

```python
result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=task_lm_callable,
    max_metric_calls=<budget>,          # total eval calls budget
    reflection_lm=reflection_lm_callable,
    reflection_minibatch_size=10,       # examples per reflection round
    reflection_prompt_template=reflection_prompt_template,  # optional
    evaluator=evaluate_callable,
    use_wandb=<True|False>,
    wandb_init_kwargs=wandb_kwargs,      # if use_wandb
    wandb_api_key=os.environ.get('WANDB_API_KEY'),  # if use_wandb
)

print("Best prompt:", result.best_candidate['system_prompt'])

# Save the optimized prompt
with open('<output_path>', 'w') as f:
    f.write(result.best_candidate['system_prompt'])
```

## Hyperparameters reference

When generating the `gepa.optimize()` call, use this reference to choose appropriate values. Ask the user about parameters marked "ask".

### Reflection tuning (most impactful)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `reflection_minibatch_size` | 3 | Number of (input, output, feedback) examples shown to the reflection LM per iteration. Higher = more signal per step but slower/more expensive. 10 is a good starting point. 15-20 if reflection LM can handle long context. Too low and reflection LM misses patterns; too high and it gets overwhelmed. |
| `candidate_selection_strategy` | `"pareto"` | Which candidate to mutate next. `"pareto"` picks from the Pareto frontier — candidates that are best on *some* subset of val examples. Lets GEPA evolve specialized candidates and merge them. `"current_best"` always mutates the globally best one (simpler, less exploration). `"epsilon_greedy"` explores randomly sometimes. `"top_k_pareto"` selects from top-k on the Pareto frontier. Default `"pareto"` is best for most cases. |
| `frontier_type` | `"instance"` | How Pareto frontier is tracked. `"instance"` = per validation example (candidate A best on examples 1,3,5; B best on 2,4). `"objective"` = per objective metric (for multi-objective). `"hybrid"` = both. `"cartesian"` = per (example, objective) pair. For single-objective tasks, use `"instance"`. |
| `skip_perfect_score` | `True` | Skip reflection when candidate scores perfectly on its minibatch (nothing to improve). Keep `True`. |
| `reflection_prompt_template` | built-in | The meta-prompt telling the reflection LM how to analyze failures and write a new prompt. Uses `<curr_param>` and `<side_info>` placeholders. Can be a dict mapping component names to templates for multi-component optimization. |
| `module_selector` | `"round_robin"` | For multi-component candidates (e.g., system prompt + output format). `"round_robin"` cycles through components. `"all"` updates all at once. Irrelevant for single-component optimization. |
| `acceptance_criterion` | `"strict_improvement"` | Whether a new candidate must be strictly better (`>`) or equal-or-better (`>=`) to be accepted. `"improvement_or_equal"` accepts ties, adding more diversity to the Pareto frontier. Worth trying for more exploration. |

### Merge (combining candidates from Pareto frontier)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `use_merge` | `False` | Enables merging two Pareto-frontier candidates into one. Idea: candidate A handles some examples well, B handles others — merge to get a candidate that handles both. Uses the reflection LM. Worth enabling when different val examples need different strategies. |
| `max_merge_invocations` | 5 | Max merge attempts. Merges are expensive (reflection LM call + full val eval). |
| `merge_val_overlap_floor` | 5 | Minimum shared val examples between two parent candidates before merge is attempted. Prevents merging candidates evaluated on disjoint subsets. |

### Budget and stopping (ask user)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `max_metric_calls` | `None` | Total evaluator calls before stopping. Each iteration costs ~`reflection_minibatch_size` train calls + val_size val calls. E.g., with minibatch=10 and 50 val examples, each iteration costs ~60 calls. **2000 calls ~ 33 iterations. 5000-10000 for serious runs.** At least one of `max_metric_calls` or `stop_callbacks` is required. |
| `stop_callbacks` | `None` | Additional/alternative stopping conditions. Can combine multiple. Available stoppers: `FileStopper` (touch a file to stop), `TimeoutStopCondition(seconds=N)`, `NoImprovementStopper(patience=N)` (stop after N iterations without improvement), `ScoreThresholdStopper(threshold=0.95)`, `SignalStopper` (Ctrl+C graceful stop), `CompositeStopper`. Import from `gepa`. |
| `perfect_score` | `1.0` | Score threshold for `skip_perfect_score`. Change if your scoring range differs. |

### Checkpointing and resumption

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `run_dir` | `None` | **Strongly recommended.** Directory for saving optimization state. If the directory exists with a prior run, GEPA resumes from the last checkpoint. Also creates a `FileStopper` — you can `touch run_dir/gepa.stop` to gracefully stop a running optimization. Set to something like `"runs/gepa_<task>_<timestamp>"`. |
| `cache_evaluation` | `False` | Caches (candidate, example) -> score. Saves metric calls if the same candidate gets re-evaluated on the same example. Enable for expensive evaluators. |

### Logging (ask user)

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `use_wandb` | `False` | Enable W&B logging. Tracks scores, candidates, iterations. |
| `wandb_init_kwargs` | `None` | Dict with `project`, `entity`, etc. |
| `wandb_api_key` | `None` | W&B API key. Usually from env var. |
| `wandb_attach_existing` | `False` | If `True`, logs to an already-active W&B run instead of creating a new one. For embedding GEPA inside a training loop. |
| `track_best_outputs` | `True` | Store the actual best outputs per val example in the result. Useful for analysis, costs memory. |
| `display_progress_bar` | `False` | tqdm progress bar over metric calls. Set `True` for interactive scripts. |
| `callbacks` | `None` | List of custom event hooks (GEPACallback protocol). Gets called on every iteration, candidate accepted/rejected, etc. For custom logging or monitoring. |

### Other

| Parameter | Default | Guidance |
|-----------|---------|----------|
| `seed` | `0` | RNG seed for reproducibility (batch sampling order). |
| `batch_sampler` | `"epoch_shuffled"` | How train examples are selected each iteration. `"epoch_shuffled"` shuffles and iterates through all examples before repeating. |
| `val_evaluation_policy` | `None` -> `"full_eval"` | Controls how many val examples are scored each iteration. `"full_eval"` scores all every time. Custom policies can subsample for speed. |
| `raise_on_exception` | `True` | If `False`, swallows evaluator/proposer errors and stops gracefully instead of crashing. Set `False` for long unattended runs. |
| `use_cloudpickle` | `False` | For serialization edge cases with dynamically generated classes. |

### Recommended defaults for typical optimization runs

For a standard single-prompt optimization run, use these defaults unless the user specifies otherwise:

```python
result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=task_lm_callable,
    reflection_lm=reflection_lm_callable,
    evaluator=evaluate_callable,
    # Reflection config
    reflection_minibatch_size=10,
    reflection_prompt_template=reflection_prompt_template,
    candidate_selection_strategy="pareto",
    # Merge — enable for tasks where different examples need different strategies
    use_merge=False,
    # Budget
    max_metric_calls=2000,
    # Checkpointing — always set for resumability
    run_dir="runs/gepa_<task_name>",
    # Logging
    display_progress_bar=True,
    use_wandb=True,  # if user wants wandb
    wandb_init_kwargs=wandb_kwargs,
    wandb_api_key=os.environ.get('WANDB_API_KEY'),
    # Reproducibility
    seed=42,
    raise_on_exception=False,  # for unattended runs
)
```

For more aggressive exploration, additionally set:
```python
    use_merge=True,
    acceptance_criterion="improvement_or_equal",
    max_metric_calls=5000,
```

## Output

1. Generate the complete script as a single `.py` file
2. Place it in `scripts/` directory (or wherever user specifies)
3. The script should be runnable with `python3 scripts/<name>.py`
4. Print the optimized prompt at the end and save it to a file
