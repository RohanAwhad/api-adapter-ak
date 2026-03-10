"""Binary correctness reward for GRPO training."""

from __future__ import annotations

import re

CORRECT_TOKEN = "CORRECT"


def extract_answer(text: str, claude_answer: int | None = None) -> int | None:
    """Extract integer answer from model output.

    If the model outputs CORRECT (the pass-through token), returns claude_answer.
    Otherwise extracts the last number from the text.
    """
    stripped = text.strip().upper()
    if CORRECT_TOKEN in stripped and claude_answer is not None:
        return claude_answer

    matches = re.findall(r"-?\d+", text)
    if matches:
        return int(matches[-1])
    return None


def correctness_reward(
    completions: list[str],
    answers: list[int],
    claude_answers: list[int | None] | None = None,
) -> list[float]:
    """Compute binary correctness reward for a batch of completions.

    Args:
        completions: Model output strings.
        answers: Ground truth integer answers.
        claude_answers: Claude's parsed answers (needed to resolve CORRECT token).

    Returns:
        List of rewards: 1.0 for correct, 0.0 for incorrect.
    """
    if claude_answers is None:
        claude_answers = [None] * len(completions)

    rewards = []
    for completion, answer, claude_ans in zip(completions, answers, claude_answers):
        predicted = extract_answer(completion, claude_answer=claude_ans)
        rewards.append(1.0 if predicted == answer else 0.0)
    return rewards
