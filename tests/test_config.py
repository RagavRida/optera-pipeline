"""Tests for optera.config — pricing, profiles, and policy."""
from __future__ import annotations

import pytest
from optera.config import cost_usd, model_info, apply_profile, CLASS_POLICY, PROFILES


class TestModelInfo:

    def test_known_model(self):
        info = model_info("claude-haiku-4-5-20251001")
        assert info["provider"] == "anthropic"
        assert info["input"] == 1.00
        assert info["output"] == 5.00

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            model_info("nonexistent-model-v99")


class TestCostUsd:
    """Cost calculation from token counts."""

    def test_haiku_simple(self):
        # 1000 input tokens @ $1/MTok + 100 output tokens @ $5/MTok
        cost = cost_usd("claude-haiku-4-5-20251001", 1000, 100)
        expected = (1000 * 1.0 / 1_000_000) + (100 * 5.0 / 1_000_000)
        assert cost == pytest.approx(expected, abs=1e-9)

    def test_opus_simple(self):
        cost = cost_usd("claude-opus-4-5-20251101", 1000, 100)
        expected = (1000 * 5.0 / 1_000_000) + (100 * 25.0 / 1_000_000)
        assert cost == pytest.approx(expected, abs=1e-9)

    def test_zero_tokens(self):
        assert cost_usd("claude-haiku-4-5-20251001", 0, 0) == 0.0

    def test_cache_write(self):
        # Cache write at 1.25x input rate for Anthropic
        cost = cost_usd("claude-haiku-4-5-20251001", 0, 0,
                         cache_write_tok=1000)
        expected = 1000 * (1.0 / 1_000_000) * 1.25
        assert cost == pytest.approx(expected, abs=1e-9)

    def test_cache_read(self):
        # Cache read at 0.10x input rate for Anthropic
        cost = cost_usd("claude-haiku-4-5-20251001", 0, 0,
                         cache_read_tok=10000)
        expected = 10000 * (1.0 / 1_000_000) * 0.10
        assert cost == pytest.approx(expected, abs=1e-9)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            cost_usd("nonexistent-model", 100, 100)

    def test_openai_model(self):
        cost = cost_usd("gpt-4o-mini", 1000, 100)
        expected = (1000 * 0.15 / 1_000_000) + (100 * 0.60 / 1_000_000)
        assert cost == pytest.approx(expected, abs=1e-9)


class TestApplyProfile:

    def test_accurate_profile(self):
        apply_profile("accurate")
        for cls in CLASS_POLICY:
            assert CLASS_POLICY[cls].model == "claude-opus-4-5-20251101"
            # Strong model = no escalation needed
            assert CLASS_POLICY[cls].escalate_to is None

    def test_cheap_profile(self):
        apply_profile("cheap")
        assert CLASS_POLICY["meter_reading"].model == "claude-haiku-4-5-20251001"
        assert CLASS_POLICY["vendor_bill"].model == "claude-haiku-4-5-20251001"
        # work_report stays at mid (Sonnet)
        assert CLASS_POLICY["work_report"].model == "claude-sonnet-4-5-20250929"

    def test_cheap_profile_has_escalation(self):
        apply_profile("cheap")
        # Cheap models should have escalation paths
        assert CLASS_POLICY["work_report"].escalate_to is not None

    def test_balanced_profile(self):
        apply_profile("balanced")
        assert CLASS_POLICY["meter_reading"].model == "claude-opus-4-5-20251101"
        assert CLASS_POLICY["vendor_bill"].model == "claude-opus-4-5-20251101"
        assert CLASS_POLICY["work_report"].model == "claude-sonnet-4-5-20250929"

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown profile"):
            apply_profile("nonexistent")

    def test_all_profiles_valid(self):
        for name in PROFILES:
            apply_profile(name)  # should not raise

    # Restore default after tests
    @pytest.fixture(autouse=True)
    def reset_policy(self):
        yield
        apply_profile("accurate")
