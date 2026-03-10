"""Tests for the custom symbol arithmetic engine."""

import random

import pytest

from api_adapter.symbols import evaluate, generate_expression


class TestEvaluate:
    def test_standard_addition(self):
        assert evaluate("3 + 5") == 8

    def test_standard_subtraction(self):
        assert evaluate("10 - 3") == 7

    def test_standard_multiplication(self):
        assert evaluate("4 * 5") == 20

    def test_standard_division(self):
        assert evaluate("20 / 4") == 5

    def test_standard_precedence_mul_before_add(self):
        assert evaluate("3 + 4 * 2") == 11

    def test_standard_precedence_div_before_sub(self):
        assert evaluate("10 - 6 / 2") == 7

    def test_standard_left_to_right(self):
        assert evaluate("10 - 3 - 2") == 5

    def test_standard_mixed_precedence(self):
        assert evaluate("2 + 3 * 4 - 6 / 2") == 11

    def test_custom_theta_addition(self):
        assert evaluate("3 θ 5") == 8

    def test_custom_alpha_subtraction(self):
        assert evaluate("10 α 3") == 7

    def test_custom_gamma_multiplication(self):
        assert evaluate("4 γ 5") == 20

    def test_custom_beta_division(self):
        assert evaluate("20 β 4") == 5

    def test_custom_precedence_gamma_before_theta(self):
        # 3 θ 4 γ 2 = 3 + (4 * 2) = 11
        assert evaluate("3 θ 4 γ 2") == 11

    def test_custom_precedence_beta_before_alpha(self):
        # 10 α 6 β 2 = 10 - (6 / 2) = 7
        assert evaluate("10 α 6 β 2") == 7

    def test_custom_mixed_precedence(self):
        # 2 θ 3 γ 4 α 6 β 2 = 2 + 12 - 3 = 11
        assert evaluate("2 θ 3 γ 4 α 6 β 2") == 11

    def test_no_spaces(self):
        assert evaluate("3+5") == 8

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            evaluate("5 / 0")


class TestGenerateExpression:
    def test_returns_tuple(self):
        expr, answer = generate_expression(2, use_custom=False, rng=random.Random(42))
        assert isinstance(expr, str)
        assert isinstance(answer, int)

    def test_custom_symbols_present(self):
        expr, _ = generate_expression(3, use_custom=True, rng=random.Random(42))
        assert any(s in expr for s in "θαγβ")

    def test_standard_symbols_present(self):
        expr, _ = generate_expression(3, use_custom=False, rng=random.Random(42))
        assert any(s in expr for s in "+-*/")
        assert not any(s in expr for s in "θαγβ")

    def test_answer_matches_evaluation(self):
        rng = random.Random(123)
        for _ in range(100):
            num_ops = rng.randint(2, 4)
            use_custom = rng.choice([True, False])
            expr, answer = generate_expression(num_ops, use_custom=use_custom, rng=rng)
            assert evaluate(expr) == answer, f"Mismatch for {expr}: expected {answer}, got {evaluate(expr)}"

    def test_integer_results(self):
        rng = random.Random(456)
        for _ in range(100):
            num_ops = rng.randint(2, 4)
            expr, answer = generate_expression(num_ops, use_custom=rng.choice([True, False]), rng=rng)
            assert answer == int(answer)

    def test_operand_count(self):
        for n in [2, 3, 4]:
            expr, _ = generate_expression(n, use_custom=False, rng=random.Random(42))
            # Count numbers in expression
            parts = expr.split()
            nums = [p for p in parts if p.isdigit()]
            assert len(nums) == n

    def test_reproducible(self):
        expr1, ans1 = generate_expression(3, use_custom=True, rng=random.Random(42))
        expr2, ans2 = generate_expression(3, use_custom=True, rng=random.Random(42))
        assert expr1 == expr2
        assert ans1 == ans2
