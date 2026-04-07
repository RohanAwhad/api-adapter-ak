_INTEGER_KWARG_FIELDS = frozenset({
    "N",
    "m",
    "n",
    "n_end",
    "n_start",
    "nth_paragraph",
    "num_bullets",
    "num_highlights",
    "num_paragraphs",
    "num_placeholders",
    "num_sections",
    "num_sentences",
    "num_words",
    "small_n",
})

def normalize_instruction_kwargs(kwargs_list):
  """Coerce integer-like instruction kwargs to ints after JSON loading."""
  normalized_kwargs = []
  for kwargs in kwargs_list:
    normalized = {}
    if kwargs is not None:
        for key, value in kwargs.items():
            if key in _INTEGER_KWARG_FIELDS and value is not None:
                normalized[key] = int(value)
            else:
                normalized[key] = value
    normalized_kwargs.append(normalized)
  return normalized_kwargs


# In[11]:


import dataclasses
from typing import Dict, Optional, Union
@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]



# In[12]:


def convert_to_input_example(example):
    ground_truth = eval(example['ground_truth'])
    return InputExample(
        key=example['key'],
        prompt=example['messages'][0]['content'],
        instruction_id_list=ground_truth[0]['instruction_id'],
        kwargs=normalize_instruction_kwargs(ground_truth[0]['kwargs'])
    )

# In[13]:


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


# In[14]:




# In[19]:


from . import instructions_registry


# In[20]:


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
  """Tests response for an upper bound for following instructions."""
  response = prompt_to_response[inp.prompt]
  if response is None:
      return OutputExample(
          instruction_id_list=inp.instruction_id_list,
          prompt=inp.prompt,
          response="",
          follow_all_instructions=False,
          follow_instruction_list=[False] * len(inp.instruction_id_list),
      )

  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )
