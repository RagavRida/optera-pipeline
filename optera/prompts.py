"""Prompt text for every stage.

Two properties matter here and they pull against each other:

1. Prompt tokens are paid on every single call, so the extraction prompt is the
   most repeated cost in the system and every clause has to earn its place.
2. Anthropic will only cache a prefix above a minimum length (2048 tokens for
   Haiku, 1024 for Sonnet/Opus). Trim a prompt below that floor and caching
   silently stops applying.

So "make the prompt shorter" is not unconditionally correct. The shared
RULEBOOK below is the cacheable prefix; per-class specs are appended after it.
"""
from __future__ import annotations

from . import fewshots, schemas

# --------------------------------------------------------------------------
# Shared rulebook. Identical bytes on every extraction call, which is what
# makes it a cache prefix rather than just boilerplate.
# --------------------------------------------------------------------------
_RULEBOOK_CORE = """You are an extraction engine for Optera, processing phone photographs of \
operational paperwork from Indian bus-fleet operators. Images arrive over WhatsApp: \
skewed, shadowed, finger-occluded, multi-lingual, and frequently handwritten.

OUTPUT CONTRACT
- Return exactly one JSON object. No prose, no markdown fence, no commentary.
- Use null for any field you cannot read. Never invent a plausible value.
- Never emit a placeholder like "N/A", "unknown", "XXXX" or "0" to stand in for
  something illegible. null is the only permitted way to say "I could not read it".

TRANSCRIPTION RULES
- Transcribe what is on the page, not what would be reasonable. If a total looks
  arithmetically wrong, still report the printed figure.
- Preserve original scripts. Gujarati and Hindi text stays in Gujarati and Hindi;
  do not translate or transliterate into Latin script.
- Preserve vehicle codes exactly: TCM35, MMM43, MAC-1, GJ-06-AV-4045.
- Indian dates are day-first. 19-06-25 is 2025-06-19. Emit YYYY-MM-DD only.
- Indian digit grouping: 1,50,000 means 150000. Strip separators in numbers.
- Strikethrough content is cancelled work, not absent work: keep the row and
  flag it.

CONFIDENCE
- "confidence" is your honest probability that a downstream clerk would accept
  this extraction without correcting it. Calibrate it: reserve values above 0.9
  for clean, fully legible documents.

REFUSAL
- If the image is a photograph of an object, a part, a vehicle, a person, a
  scene, or is blank or unreadable, it carries no structured record. Do not
  manufacture one. Refuse it as described in the task section below.
"""

# --------------------------------------------------------------------------
# RULEBOOK variants:
#
# RULEBOOK      — core rules only (~400 tokens). Used when caching is
#                 not beneficial (e.g. single-shot calls, short jobs).
#
# RICH_RULEBOOK — core + all six few-shot examples (~2100 tokens). This
#                 exceeds both the Haiku (2048) and Sonnet/Opus (1024)
#                 prompt-caching floors, so the system prompt is cached
#                 after the first call. The ~$0.013 cache-write cost is
#                 recovered after ~2 subsequent reads.
#
#                 The examples also directly improve accuracy on the three
#                 failure modes they demonstrate: decimal odometers, blank
#                 checklist rows, and the DEF-label refusal trap.
# --------------------------------------------------------------------------
RULEBOOK = _RULEBOOK_CORE

# Lazily constructed so fewshots.py is imported only when needed.
_RICH_RULEBOOK: str | None = None


def rich_rulebook() -> str:
    global _RICH_RULEBOOK
    if _RICH_RULEBOOK is None:
        _RICH_RULEBOOK = _RULEBOOK_CORE + "\n\n" + fewshots.all_examples()
    return _RICH_RULEBOOK


def classification_prompt() -> str:
    """Router prompt. Terse because it is paired with a thumbnail and one word of output."""
    lines = [f"- {name}: {desc}" for name, desc in schemas.CLASS_DESCRIPTIONS.items()]
    return (
        "Classify this photograph into exactly one class.\n\n"
        + "\n".join(lines)
        + "\n\nJudge by layout and purpose, not by legibility - the image is a "
          "deliberately small thumbnail and you are not expected to read the text.\n"
          "A page of ruled handwriting is work_report. A printed letterhead with "
          "amounts is vendor_bill. A lit display of numbers is meter_reading. "
          "A physical object is not_a_document.\n\n"
        "If and only if the class is meter_reading, also set subtype to "
        '"odometer" (a vehicle instrument cluster) or "dispenser" (a fuel/DEF '
        "pump display). Otherwise set subtype to null.\n\n"
        'Reply with only: {"class":"<class>","subtype":<string|null>,"confidence":<0-1>}'
    )


# Guidance targeted at failure modes actually observed in evaluation, not
# imagined ones. Each line here traces to a specific wrong extraction.
CLASS_HINTS: dict[str, str] = {
    "work_report": schemas.BUS_CODE_HINT + (
        "\nThe date is usually handwritten into a partly pre-printed field: the "
        "operator writes the day and month and the year is already printed. Read "
        "the handwritten digits, not the printed year, when they conflict."
    ),
    "meter_reading": (
        "A vehicle instrument cluster shows several numbers that are NOT "
        "interchangeable:\n"
        "- ODO is cumulative distance; TRIP A/B is a resettable counter. Do not swap them.\n"
        "- AFE / 'km/l' is average fuel efficiency. It is NOT a price and must never "
        "go into rate_per_litre.\n"
        "- A clock such as 08:21 AM is a time of day, not a date. Do not invent a "
        "calendar date from it. Only fill captured_at from a visible date overlay.\n"
        "- Odometer values are frequently shown to one decimal place. 32065.4 and "
        "320654 are different readings; transcribe the decimal point if it is there."
    ),
    "vendor_bill": (
        "Bill numbers and vehicle registrations are the fields most often misread. "
        "Read each character of the registration separately - 0/6, 1/6, 8/9 and 5/6 "
        "are the usual confusions. If a field on the pad was left blank, return null; "
        "a blank line on a printed form is not a zero."
    ),
}


def extraction_prompt(doc_class: str, quality_flags: list[str] | None = None,
                      subtype: str | None = None) -> str:
    """Per-class task text appended after the cached rulebook."""
    spec = schemas.prompt_spec(doc_class, subtype=subtype)
    hint = ""
    if quality_flags:
        readable = {
            "low_sharpness": "This image is out of focus; be conservative and prefer null over a guessed digit.",
            "underexposed": "This image is dark; glare and shadow may hide digits.",
            "truncated_file_recovered": "This file was truncated in transit; the lower portion may be missing or corrupted.",
            "very_low_resolution": "This image is very low resolution.",
        }
        notes = [readable[f] for f in quality_flags if f in readable]
        if notes:
            hint = "\nIMAGE QUALITY WARNING: " + " ".join(notes) + "\n"

    guidance = CLASS_HINTS.get(doc_class, "")
    return f"""TASK: extract this {doc_class} into the schema below.
{hint}
{guidance}

SCHEMA
{spec}

Return this exact envelope:
{{"doc_class":"{doc_class}","confidence":<0-1>,"data":{{...schema fields...}}}}

If this image is NOT a {doc_class} after all, ignore the schema and return:
{{"doc_class":"<the correct class>","confidence":<0-1>,"data":null,
  "refusal":{{"reason":"misrouted","observed":"<what it actually shows>"}}}}"""


def baseline_prompt() -> str:
    """The naive 1x approach: one big prompt, every schema, one big model, full image.

    This is intentionally what a competent engineer writes on day one before
    thinking about cost - not a strawman. It gets the same rulebook, the same
    schemas and the same refusal contract as the optimised path, so the
    comparison isolates cost engineering rather than prompt quality.
    """
    all_specs = "\n\n".join(schemas.prompt_spec(c, compact=False) for c in schemas.CLASS_FIELDS)
    class_list = "\n".join(f"- {n}: {d}" for n, d in schemas.CLASS_DESCRIPTIONS.items())
    return f"""TASK: identify what this photograph is, then extract it.

CLASSES
{class_list}

SCHEMAS
{all_specs}

Return exactly:
{{"doc_class":"<class>","confidence":<0-1>,"data":{{...fields for that class...}}}}

If the image is not_a_document, return:
{{"doc_class":"not_a_document","confidence":<0-1>,"data":null,
  "refusal":{{"reason":"not_a_document","observed":"<what it shows>"}}}}"""
