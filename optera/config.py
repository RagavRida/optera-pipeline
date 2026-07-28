"""OpenAI model configuration and auditable token pricing."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PRICING_PATH = Path(__file__).resolve().parent / "pricing.json"
with open(PRICING_PATH) as fh:
    MODELS: dict[str, dict] = json.load(fh)["models"]


def model_info(model: str) -> dict:
    if model not in MODELS:
        raise KeyError(
            f"Unknown model {model!r}. Add its per-MTok rates to optera/pricing.json."
        )
    return MODELS[model]


def cost_usd(model: str, in_tok: int, out_tok: int,
             cache_write_tok: int = 0, cache_read_tok: int = 0) -> float:
    """Compute a call's USD cost from API-reported usage."""
    m = model_info(model)
    input_rate = m["input"] / 1_000_000
    return (
        in_tok * input_rate
        + out_tok * m["output"] / 1_000_000
        + cache_write_tok * input_rate * m.get("cache_write", 1.0)
        + cache_read_tok * input_rate * m.get("cache_read", 1.0)
    )


# The committed benchmark uses GPT-4o as the big baseline and extractor, then
# GPT-4o-mini only for inexpensive 384px triage.
BASELINE_MODEL = os.getenv("OPTERA_BASELINE_MODEL", "gpt-4o")
ROUTER_MODEL = os.getenv("OPTERA_ROUTER_MODEL", "gpt-4o-mini")
EXTRACT_MODEL = os.getenv("OPTERA_EXTRACT_MODEL", "gpt-4o")

BASELINE_MAX_DIM = 1568
BASELINE_JPEG_Q = 90
ROUTER_MAX_DIM = int(os.getenv("OPTERA_ROUTER_MAX_DIM", "384"))
ROUTER_JPEG_Q = 70
MAX_USEFUL_DIM = 1568
EXTRACT_JPEG_Q = 92


@dataclass(frozen=True)
class ClassPolicy:
    model: str
    max_dim: int
    jpeg_q: int
    max_tokens: int


# Class-specific schemas are retained because they reduce prompt/output clutter;
# the benchmark uses the same high-accuracy extractor for every real document.
CLASS_POLICY: dict[str, ClassPolicy] = {
    "meter_reading": ClassPolicy(EXTRACT_MODEL, 1568, EXTRACT_JPEG_Q, 400),
    "vendor_bill": ClassPolicy(EXTRACT_MODEL, 1568, EXTRACT_JPEG_Q, 1600),
    "work_report": ClassPolicy(EXTRACT_MODEL, 1568, EXTRACT_JPEG_Q, 3000),
}

PHASH_THRESHOLD = int(os.getenv("OPTERA_PHASH_THRESHOLD", "5"))
BLUR_VAR_FLOOR = 150.0
DARK_MEAN_FLOOR = 90.0
DEGRADED_RES_BOOST = 1.35
VALID_CLASSES = ("work_report", "vendor_bill", "meter_reading", "not_a_document")
