"""Claude API client for arithmetic evaluation via Google Vertex AI."""

from __future__ import annotations

import asyncio
import os
import re

from anthropic import AnthropicVertex, AsyncAnthropicVertex


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_PROJECT = ""  # Set via GOOGLE_CLOUD_PROJECT env var
DEFAULT_REGION = ""   # Set via GOOGLE_CLOUD_REGION env var
DEFAULT_CONCURRENCY = 30

SYSTEM_PROMPT = (
    "You are an arithmetic calculator. "
    "Evaluate the given expression and respond with ONLY the numerical answer. "
    "No explanation, no working, just the number."
)


def get_client() -> AnthropicVertex:
    """Create Anthropic Vertex AI client using GCP Application Default Credentials."""
    return AnthropicVertex(
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT),
        region=os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION),
    )


def get_async_client() -> AsyncAnthropicVertex:
    """Create async Anthropic Vertex AI client."""
    return AsyncAnthropicVertex(
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT),
        region=os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION),
    )


def query_claude(
    expression: str,
    client: AnthropicVertex | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send an arithmetic expression to Claude (sync)."""
    if client is None:
        client = get_client()

    message = client.messages.create(
        model=model,
        max_tokens=64,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Evaluate: {expression}"}],
    )
    return message.content[0].text.strip()


async def _query_one(
    sample: dict,
    client: AsyncAnthropicVertex,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Query Claude for a single sample with concurrency control."""
    async with semaphore:
        message = await client.messages.create(
            model=model,
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Evaluate: {sample['expression']}"}],
        )
        response = message.content[0].text.strip()
        parsed = parse_answer(response)
        return {
            **sample,
            "claude_response": response,
            "claude_answer": parsed,
            "claude_correct": parsed == sample["answer"],
        }


def parse_answer(response: str) -> int | None:
    """Extract integer answer from Claude's response."""
    match = re.search(r"-?\d+", response)
    if match:
        return int(match.group())
    return None


async def _run_baseline_async(
    samples: list[dict],
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """Run baseline with concurrent API calls."""
    client = get_async_client()
    semaphore = asyncio.Semaphore(concurrency)

    done_count = 0
    total = len(samples)

    async def tracked_query(sample: dict) -> dict:
        nonlocal done_count
        result = await _query_one(sample, client, model, semaphore)
        done_count += 1
        if done_count % 100 == 0 or done_count == total:
            print(f"[{done_count}/{total}] completed")
        return result

    results = await asyncio.gather(*[tracked_query(s) for s in samples])
    return list(results)


def run_baseline(
    samples: list[dict],
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    **_kwargs,
) -> list[dict]:
    """Run all samples through Claude in parallel.

    Args:
        samples: List of {"expression": str, "answer": int, "type": str}.
        model: Model to use.
        concurrency: Max concurrent API calls.

    Returns:
        List of dicts with added "claude_response" and "claude_answer" fields.
    """
    print(f"Running {len(samples)} samples through {model} (concurrency={concurrency})")
    results = asyncio.run(_run_baseline_async(samples, model=model, concurrency=concurrency))

    correct = sum(1 for r in results if r["claude_correct"])
    print(f"Final accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")
    return results
