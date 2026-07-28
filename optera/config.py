"""Central configuration: pricing, model tiers, and per-class routing policy.

Every tunable that affects cost or accuracy lives here so that the cost/accuracy
tradeoff is a config decision you can audit, not a magic number buried in a prompt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICING_PATH = Path(__file__).resolve().parent / "pricing.json"

with open(PRICING_PATH) as fh:
    _PRICING = json.load(fh)

MODELS: dict[str, dict] = _PRICING["models"]


def model_info(model: str) -> dict:
    if model not in MODELS:
        raise KeyError(
            f"Unknown model {model!r}. Add it to optera/pricing.json with its "
            f"per-MTok rates so its cost can be accounted for."
        )
    return MODELS[model]


def cost_usd(model: str, in_tok: int, out_tok: int,
             cache_write_tok: int = 0, cache_read_tok: int = 0) -> float:
    """Exact USD for one call, from API-reported token counts.

    Anthropic reports cache_creation/cache_read separately from input_tokens, so
    the three buckets are additive and must not be double counted.
    """
    m = model_info(model)
    rate_in = m["input"] / 1_000_000
    rate_out = m["output"] / 1_000_000
    return (
        in_tok * rate_in
        + out_tok * rate_out
        + cache_write_tok * rate_in * m.get("cache_write", 1.0)
        + cache_read_tok * rate_in * m.get("cache_read", 1.0)
    )


# --------------------------------------------------------------------------
# Which concrete models fill each role. Overridable by env so a reviewer can
# re-run the whole comparison against different vendors without editing code.
# --------------------------------------------------------------------------
BASELINE_MODEL = os.getenv("OPTERA_BASELINE_MODEL", "claude-opus-4-5-20251101")
ROUTER_MODEL = os.getenv("OPTERA_ROUTER_MODEL", "claude-haiku-4-5-20251001")
CHEAP_MODEL = os.getenv("OPTERA_CHEAP_MODEL", "claude-haiku-4-5-20251001")
MID_MODEL = os.getenv("OPTERA_MID_MODEL", "claude-sonnet-4-5-20250929")
STRONG_MODEL = os.getenv("OPTERA_STRONG_MODEL", "claude-opus-4-5-20251101")

# Baseline deliberately mirrors the naive approach: full-resolution image,
# one big model, one call, no routing. This is the 1x we have to beat.
BASELINE_MAX_DIM = 1568       # Anthropic's own recommended long edge
BASELINE_JPEG_Q = 90

# Router sees only a thumbnail. Classification needs layout, not legibility.
ROUTER_MAX_DIM = int(os.getenv("OPTERA_ROUTER_MAX_DIM", "384"))
ROUTER_JPEG_Q = 70


@dataclass
class ClassPolicy:
    """Extraction policy for one document class.

    max_dim is the empirically derived *resolution floor* - the smallest long
    edge at which this class still extracts correctly. See scripts/sweep_resolution.py;
    the floors are measured against ground truth, not guessed.
    """
    model: str
    max_dim: int
    jpeg_q: int
    max_tokens: int
    escalate_to: str | None = None
    notes: str = ""


# Per-class policy. The whole cost thesis is in this table: a dashboard photo
# carries ~4 numbers and needs a small model at low resolution, while a dense
# multilingual handwritten table needs real capacity and real pixels.
#
# EVERY VALUE BELOW COMES FROM scripts/sweep_resolution.py, NOT FROM INTUITION.
# Measured accuracy on the gold subset (see out/sweep.json):
#
#   meter_reading   haiku  768/1024/1568 -> 57.1% / 64.3% / 71.4%
#                   sonnet 768/1024/1568 -> 71.4% / 57.1% / 64.3%
#     => resolution matters more than model here; haiku@1568 ties the best
#        result at a third of the price.
#
#   vendor_bill     haiku  1024/1568 -> 57.1% / 59.5%
#                   sonnet 1024/1568 -> 59.5% / 61.9%
#     => sonnet buys 2.4 points for 3x the money. Take haiku and let the
#        arithmetic validator escalate the ones that are actually wrong.
#
#   work_report     haiku  1024/1568 -> 23.3% / 26.7%
#                   sonnet 1024/1568 -> 26.7% / 56.7%
#     => a genuine capability cliff. Haiku cannot read these pages at any
#        resolution, and sonnet needs the full 1568px.
#
# THE SWEEP ABOVE HAD A BLIND SPOT: it only compared haiku against sonnet.
# Scoring the Opus baseline per class exposed the opposite of what the sweep
# implied:
#
#   class            opus     cheap-tier    delta
#   meter_reading     95%        79%       -15.8
#   vendor_bill       81%        60%       -21.4
#   work_report       63%        57%        -6.7
#
# Opus is far ahead precisely on the two classes that are CHEAP to run it on -
# a dashboard photo is one small image and ~110 output tokens, an invoice ~400 -
# and barely ahead on the one class that is expensive, where 25 dense pages
# dominate both volume and output tokens.
#
# So the policy is the inverse of the intuitive one: buy the best model where
# it is cheap and accuracy-critical, and economise only where the tokens
# actually are. Cost per class is not proportional to how hard the class is.
#
MAX_USEFUL_DIM = 1568  # vendor downscales beyond this, so more pixels cost encode time for zero benefit

# MEASURED: JPEG quality does not change token cost at all. The same 1568px
# invoice bills 1585 input tokens at q=70 (120 KB) and at q=95 (326 KB) -
# vision billing is a function of pixel dimensions, not payload bytes.
#
# So "compress the image harder" is a fake optimisation: it destroys detail in
# small printed text like GSTINs and saves nothing. Only downscaling saves
# money. We therefore encode at high quality everywhere and treat resolution as
# the sole image-side cost lever.
EXTRACT_JPEG_Q = 92

CLASS_POLICY: dict[str, ClassPolicy] = {
    "meter_reading": ClassPolicy(
        model=STRONG_MODEL, max_dim=1568, jpeg_q=EXTRACT_JPEG_Q, max_tokens=400,
        escalate_to=None,
        notes="~110 output tokens, so the best model costs about $0.010/doc here. "
              "Every digit is load-bearing and there is no internal arithmetic to "
              "validate an odometer against, so a silent misread is unrecoverable. "
              "Cheapest place in the whole pipeline to buy accuracy.",
    ),
    "vendor_bill": ClassPolicy(
        model=STRONG_MODEL, max_dim=1568, jpeg_q=EXTRACT_JPEG_Q, max_tokens=1600,
        escalate_to=None,
        notes="Financial values: a wrong total is the most damaging error the "
              "system can make. +21 points for roughly $0.015/doc on 10 documents. "
              "Arithmetic validation still runs as a safety net.",
    ),
    "work_report": ClassPolicy(
        model=MID_MODEL, max_dim=1568, jpeg_q=EXTRACT_JPEG_Q, max_tokens=3000,
        escalate_to=STRONG_MODEL,
        notes="The volume class: 25 of 47 documents and 80%+ of all output tokens. "
              "Opus buys only ~6.7 points here (within the noise of a 3-document "
              "gold set) for roughly double the price, so this is where we "
              "economise - Sonnet plus compact row encoding, with escalation "
              "reserved for documents that fail deterministic validation.",
    ),
}

# Confidence below which we spend more money on a second, stronger pass.
ESCALATION_THRESHOLD = float(os.getenv("OPTERA_ESCALATION_THRESHOLD", "0.55"))

# Perceptual-hash Hamming distance under which two images are treated as the
# same document. WhatsApp re-encodes on every forward, so exact hashing misses
# real duplicates; 5 is conservative enough to avoid collapsing sibling pages.
PHASH_THRESHOLD = int(os.getenv("OPTERA_PHASH_THRESHOLD", "5"))

# Quality gates. Calibrated against the starter set's actual distribution
# (median sharpness 473; the one visibly out-of-focus odometer scores 105, and
# night dashboard shots sit near 82 luminance). These do not reject anything -
# they raise the resolution floor for that document, so we spend more pixels
# exactly where the image is worst instead of uniformly everywhere.
BLUR_VAR_FLOOR = 150.0    # variance of Laplacian
DARK_MEAN_FLOOR = 90.0    # mean luminance 0-255
DEGRADED_RES_BOOST = 1.35  # multiply the class resolution floor when flagged

VALID_CLASSES = ("work_report", "vendor_bill", "meter_reading", "not_a_document")


# --------------------------------------------------------------------------
# Operating points.
#
# There is no single right answer to "how cheap should this be" - it depends
# on what a wrong field costs the business. Rather than hardcode one tradeoff,
# the pipeline exposes the measured frontier and lets the operator choose.
# --------------------------------------------------------------------------
PROFILES: dict[str, dict[str, str]] = {
    # Maximum savings. Small models everywhere, validator-driven escalation as
    # the only safety net. Appropriate when a human reviews the output anyway.
    "cheap": {"meter_reading": CHEAP_MODEL, "vendor_bill": CHEAP_MODEL,
              "work_report": MID_MODEL},
    # Best model on the two classes where it is cheap to run and where errors
    # are financial; economise on the high-volume handwriting class. Cheaper
    # than "accurate" but measurably less accurate on work reports.
    "balanced": {"meter_reading": STRONG_MODEL, "vendor_bill": STRONG_MODEL,
                 "work_report": MID_MODEL},
    # DEFAULT. Measured accuracy parity with the naive baseline (80.2% both),
    # yet 1.52x cheaper, because routing, deduplication, free rejects, narrowed
    # schemas and compact output encoding all still apply.
    #
    # This profile DOMINATES the other two on this corpus: it costs only 19%
    # more than "cheap" while scoring 20 points higher. Model downgrading is a
    # trap here - the spend is concentrated in work-report OUTPUT tokens, which
    # no profile changes. Keep "cheap"/"balanced" for corpora with a different
    # class mix, where downgrading would actually move the bill.
    "accurate": {"meter_reading": STRONG_MODEL, "vendor_bill": STRONG_MODEL,
                 "work_report": STRONG_MODEL},
}


def apply_profile(name: str) -> None:
    """Repoint CLASS_POLICY at the models for the named operating point."""
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}")
    for cls, model in PROFILES[name].items():
        CLASS_POLICY[cls].model = model
        # Escalation only makes sense while there is a stronger model to buy.
        CLASS_POLICY[cls].escalate_to = (
            None if model == STRONG_MODEL
            else (STRONG_MODEL if cls == "work_report" else MID_MODEL))
