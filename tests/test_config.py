"""Tests for OpenAI model pricing and the benchmark policy."""
from __future__ import annotations

import pytest

from optera.config import BASELINE_MODEL, CLASS_POLICY, ROUTER_MODEL, cost_usd, model_info


def test_known_model_is_openai():
    assert model_info("gpt-4o-mini")["provider"] == "openai"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        model_info("nonexistent-model-v99")


def test_openai_cost_includes_cached_prompt_tokens_at_discount():
    cost = cost_usd("gpt-4o-mini", 1_000, 100, cache_read_tok=2_000)
    expected = (1_000 * 0.15 + 100 * 0.60 + 2_000 * 0.15 * 0.50) / 1_000_000
    assert cost == pytest.approx(expected)


def test_benchmark_model_roles_are_openai():
    assert BASELINE_MODEL == "gpt-4o"
    assert ROUTER_MODEL == "gpt-4o-mini"
    assert {policy.model for policy in CLASS_POLICY.values()} == {"gpt-4o"}
