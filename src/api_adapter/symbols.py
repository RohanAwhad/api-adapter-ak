"""Custom symbol arithmetic engine.

Symbol mapping:
    θ (theta) → addition (+)
    α (alpha) → subtraction (-)
    γ (gamma) → multiplication (×)
    β (beta)  → division (÷)

Precedence (BODMAS): β, γ (higher) before θ, α (lower). Left-to-right within same level.
"""

from __future__ import annotations

import random

CUSTOM_SYMBOLS = {"θ": "+", "α": "-", "γ": "*", "β": "/"}
STANDARD_OPS = ["+", "-", "*", "/"]
HIGH_PRECEDENCE = {"*", "/"}
LOW_PRECEDENCE = {"+", "-"}

SYMBOL_DEFINITIONS = (
    "Custom arithmetic symbols:\n"
    "  θ means addition (+)\n"
    "  α means subtraction (-)\n"
    "  γ means multiplication (×)\n"
    "  β means division (÷)\n"
    "Precedence: β and γ bind tighter than θ and α (same as standard BODMAS)."
)


def _tokenize(expr: str) -> list[str]:
    """Tokenize an expression into numbers and operators."""
    tokens: list[str] = []
    current = ""
    for ch in expr.replace(" ", ""):
        if ch.isdigit():
            current += ch
        else:
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
    if current:
        tokens.append(current)
    return tokens


def _to_standard(tokens: list[str]) -> list[str]:
    """Replace custom symbols with standard operators."""
    return [CUSTOM_SYMBOLS.get(t, t) for t in tokens]


def evaluate(expr: str) -> int:
    """Evaluate an arithmetic expression (custom or standard symbols).

    Handles operator precedence (×/÷ before +/-), left-to-right.
    Returns integer result.
    """
    tokens = _to_standard(_tokenize(expr))

    nums: list[float] = []
    ops: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i].lstrip("-").isdigit():
            nums.append(float(tokens[i]))
        else:
            ops.append(tokens[i])
        i += 1

    # First pass: evaluate high-precedence ops (*, /)
    new_nums: list[float] = [nums[0]]
    new_ops: list[str] = []
    for j, op in enumerate(ops):
        if op in HIGH_PRECEDENCE:
            left = new_nums.pop()
            right = nums[j + 1]
            if op == "*":
                new_nums.append(left * right)
            else:
                new_nums.append(left / right)
        else:
            new_nums.append(nums[j + 1])
            new_ops.append(op)

    # Second pass: evaluate low-precedence ops (+, -)
    result = new_nums[0]
    for j, op in enumerate(new_ops):
        if op == "+":
            result += new_nums[j + 1]
        else:
            result -= new_nums[j + 1]

    return int(result)


def generate_expression(
    num_operands: int,
    use_custom: bool,
    rng: random.Random | None = None,
) -> tuple[str, int]:
    """Generate a random arithmetic expression with integer result.

    Args:
        num_operands: Number of operands (2-4).
        use_custom: If True, use custom symbols (θ, α, γ, β). Otherwise standard (+, -, *, /).
        rng: Random instance for reproducibility.

    Returns:
        (expression_string, correct_answer)
    """
    if rng is None:
        rng = random.Random()

    custom_ops = list(CUSTOM_SYMBOLS.keys())
    max_attempts = 1000

    for _ in range(max_attempts):
        operands = [rng.randint(1, 99) for _ in range(num_operands)]
        if use_custom:
            operators = [rng.choice(custom_ops) for _ in range(num_operands - 1)]
        else:
            operators = [rng.choice(STANDARD_OPS) for _ in range(num_operands - 1)]

        # Build expression string
        parts = [str(operands[0])]
        for k in range(len(operators)):
            parts.append(operators[k])
            parts.append(str(operands[k + 1]))
        expr = " ".join(parts)

        try:
            result = evaluate(expr)
        except (ZeroDivisionError, ValueError):
            continue

        # Check integer result
        # Re-evaluate with floats to make sure it's truly integer
        tokens = _to_standard(_tokenize(expr))
        nums_f: list[float] = []
        ops_f: list[str] = []
        for t in tokens:
            if t.lstrip("-").isdigit():
                nums_f.append(float(t))
            else:
                ops_f.append(t)

        # Evaluate with float precision
        new_nums_f: list[float] = [nums_f[0]]
        new_ops_f: list[str] = []
        valid = True
        for j, op in enumerate(ops_f):
            if op in HIGH_PRECEDENCE:
                left = new_nums_f.pop()
                right = nums_f[j + 1]
                if op == "/" and right == 0:
                    valid = False
                    break
                val = left * right if op == "*" else left / right
                if op == "/" and val != int(val):
                    valid = False
                    break
                new_nums_f.append(val)
            else:
                new_nums_f.append(nums_f[j + 1])
                new_ops_f.append(op)

        if not valid:
            continue

        float_result = new_nums_f[0]
        for j, op in enumerate(new_ops_f):
            if op == "+":
                float_result += new_nums_f[j + 1]
            else:
                float_result -= new_nums_f[j + 1]

        if float_result != int(float_result):
            continue

        return expr, int(float_result)

    raise RuntimeError(f"Failed to generate valid expression after {max_attempts} attempts")
