"""
CUDA_VISIBLE_DEVICES=2 python scripts/ifbench/gepa_optimize_adapter_system_prompt.py
"""

from datasets import Dataset

dataset = Dataset.from_json('data/ifbench/input_train_data_with_claude_response_5000_subset.jsonl')
dataset = dataset.shuffle(seed=42).select(range(150))
dataset


# # baseline

import os
from unsloth import FastLanguageModel


import re
ptrn = re.compile(r"<\|ADAPTER_RESPONSE_START\|>(.*)<\|ADAPTER_RESPONSE_END\|>", re.DOTALL)


SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to look at the user prompt and the draft response and output <|ADAPTER_RESPONSE_START|>CORRECT<|ADAPTER_RESPONSE_END|> if the draft response is correct
If the draft response is incorrect, print the correct final answer wrapped in <|ADAPTER_RESPONSE_START|> ... <|ADAPTER_RESPONSE_END|> tags.
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
DEFAULT_MAX_SEQ_LENGTH = 32768
DEFAULT_LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_MODEL_NAME,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bf16 on H100)
    load_in_4bit=False,
    gpu_memory_utilization=0.5,
    fast_inference=True,
)

FastLanguageModel.for_inference(model)




eval_dataset = dataset.select(range(50))
train_dataset = dataset.select(range(50, 150))
eval_dataset, train_dataset


texts = [
    tokenizer.apply_chat_template(
        x['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    for x in eval_dataset
]

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=16384,
)


outputs = model.fast_generate(
    texts,
    sampling_params=sampling_params,
)

all_outputs = [o.outputs[0].text for o in outputs]
print(f"Inference complete.")

from api_adapter.ifbench.eval_utils import (
    test_instruction_following_loose,
    InputExample,
    normalize_instruction_kwargs
)

def reward_fn_with_feedback(prompts, completions, ground_truth, key, claude_reward, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    # extract the boxed answer from each response
    rewards, feedbacks = [], []
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
                if adapter_response == 'CORRECT':
                    rewards.append(1.0)
                    feedbacks.append("The draft response was correct. And the adapter output 'CORRECT'.")
                else:
                    rewards.append(0.0)
                    feedbacks.append("The draft response was correct. But the adapter didn't output 'CORRECT'.")
                continue
            prompt_to_response = {inp.prompt: adapter_response}
            r = test_instruction_following_loose(inp, prompt_to_response)
            rewards.append(float(r.follow_all_instructions))

            # format the feedback
            feedback = ""
            for instruction_id, did_follow in zip(r.instruction_id_list, r.follow_instruction_list):
                if did_follow:
                    feedback += f"Followed instruction: {instruction_id}\n"
                else:
                    feedback += f"Did not follow instruction: {instruction_id}\n"
            feedbacks.append(feedback)
        except Exception as e:
            # logger.exception(e)
            rewards.append(0.0)
            # format the feedback
            feedback = (
                f"Received an error while calculating the reward.\nError: {e}"
                "\n- If 'list index out of range' error, it means the function was unable to find the adapter response inside the tags."
            )
            feedbacks.append(feedback)
    return rewards, feedbacks



tmp = eval_dataset[:]
prompts = tmp['prompt']
completions = [
    [{'content': r}]
    for r in all_outputs
]

ground_truth = tmp['ground_truth']
key = tmp['key']
claude_reward = tmp['claude_reward']

rewards, _ = reward_fn_with_feedback(prompts, completions, ground_truth, key, claude_reward)
print('Baseline reward:', sum(rewards) / len(rewards))


# GEPA optimization

import gepa


# trainset is a list of dict.
# dict has keys: 'input' and 'answer'

trainset = []
for x in train_dataset:
    inp = x['prompt'][-1]['content']
    ans = {'ground_truth': x['ground_truth'], 'key': x['key'], 'claude_reward': x['claude_reward']}
    trainset.append({'input': inp, 'answer': ans})

valset = []
for x in eval_dataset:
    inp = x['prompt'][-1]['content']
    ans = {'ground_truth': x['ground_truth'], 'key': x['key'], 'claude_reward': x['claude_reward']}
    valset.append({'input': inp, 'answer': ans})

seed_prompt = {"system_prompt": SYSTEM_PROMPT}

# task lm callable
'''
class ChatCompletionCallable(Protocol):
    """Protocol for chat completion callables (duck typing for custom model wrappers)."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...
'''
from vllm import SamplingParams

def task_lm_callable(messages) -> str:
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    ]

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=8192,
    )

    outputs = model.fast_generate(
        texts,
        sampling_params=sampling_params,
    )

    all_outputs = [o.outputs[0].text for o in outputs]
    return all_outputs[0]


x = task_lm_callable([{"role": "user", "content": "What is the capital of France?"}])
print(f'Task LM callable: {x}')


import os
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
os.environ["GOOGLE_CLOUD_REGION"] = os.environ["CLOUD_ML_REGION"]
# Generate api completions
from anthropic import AnthropicVertex

client = AnthropicVertex(
    region=os.environ["GOOGLE_CLOUD_REGION"],
    project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
)


def reflection_lm_callable(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, str): prompt = [{"role": "user", "content": prompt}]
    # get system prompt
    final_messages_list = []
    system_prompt = None
    for m in prompt:
        if m['role'] == 'system':
            if system_prompt is None: system_prompt = m['content']
        else:
            final_messages_list.append(m)

    kwargs = {
        "model": "claude-opus-4-6",
        "max_tokens": 64000,
        "messages": final_messages_list,
        "thinking": {"type": "adaptive"},
        "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
    }
    if system_prompt is not None:
        kwargs["system"] = system_prompt

    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    return response.content[-1].text

print('Reflection LM callable:', reflection_lm_callable("What is the capital of France?"))

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

Let the assistant think before generating the response in the specified tags of adapter response start and end. 

Provide the new instructions within ``` blocks."""


# evaluate function
'''
# Callable that evaluates a response and returns (score, feedback, optional objective_scores)
class Evaluator(Protocol):
    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        """
        Evaluates a response and returns a score, feedback, and optional objective scores.
        """
        ...
'''

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

def evaluate_callable(data, response) -> EvaluationResult:
    prompt = data['input']
    messages = [{'role': 'user', 'content': prompt}]
    completion = [{'content': response}]
    ans = data['answer']
    ground_truth = ans['ground_truth']
    key = ans['key']
    claude_reward = ans['claude_reward']
    rewards, feedbacks = reward_fn_with_feedback([messages], [completion], [ground_truth], [key], [claude_reward])
    return EvaluationResult(
        score=rewards[0],
        feedback=feedbacks[0],
        objective_scores=None,
    )

import dotenv
dotenv.load_dotenv(override=True)
if os.environ.get('WANDB_API_KEY') is None: print("WANDB_API_KEY is not set")
wandb_kwargs = {'project': 'api-adapter-ifbench-gepa', 'entity': 'ronny21'}
os.environ['WANDB_RUN_NAME'] = 'gepa-optimize-v2-system-prompt'


from pathlib import Path
CHECKPOINT_DIR = Path(f'outputs/ifbench/gepa_optimize_adapter_system_prompt/{os.environ["WANDB_RUN_NAME"]}')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=task_lm_callable,
    max_metric_calls=50000,
    reflection_lm=reflection_lm_callable,
    reflection_minibatch_size=5,
    reflection_prompt_template=reflection_prompt_template,
    evaluator=evaluate_callable,
    use_merge=True,
    max_merge_invocations=50000,
    acceptance_criterion="improvement_or_equal",
    run_dir=CHECKPOINT_DIR,
    # logging args
    use_wandb=True,
    wandb_init_kwargs=wandb_kwargs,
    wandb_api_key=os.environ['WANDB_API_KEY'],
)

print("Optimized prompt:", result.best_candidate['system_prompt'])

with open(CHECKPOINT_DIR / 'optimized_system_prompt.txt', 'w') as f:
    f.write(result.best_candidate['system_prompt'])

eval_dataset = eval_dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": result.best_candidate['system_prompt']},
        {"role": "user", "content": x['prompt'][-1]['content']}
    ]
})

texts = [
    tokenizer.apply_chat_template(x['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for x in eval_dataset
]

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=8192,
)

outputs = model.fast_generate(
    texts,
    sampling_params=sampling_params,
)

all_outputs = [o.outputs[0].text for o in outputs]


tmp = eval_dataset[:]
prompts = tmp['prompt']
completions = [
    [{'content': r}]
    for r in all_outputs
]

ground_truth = tmp['ground_truth']
key = tmp['key']
claude_reward = tmp['claude_reward']

rewards, _ = reward_fn_with_feedback(prompts, completions, ground_truth, key, claude_reward)
print(f"GEPA optimized reward: {sum(rewards) / len(rewards)}")
