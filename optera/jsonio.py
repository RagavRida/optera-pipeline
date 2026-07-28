"""Tolerant JSON recovery from model output.

Models occasionally wrap JSON in a markdown fence, prepend a sentence, or get
truncated by max_tokens mid-object. Local recovery preserves usable output
without spending another model call.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _balance(s: str) -> str:
    """Close unterminated strings/objects/arrays from a truncated response.

    A response cut off by max_tokens is usually 95% good data; discarding it
    means paying twice for the same document.
    """
    out, stack, in_str, esc = [], [], False, False
    for ch in s:
        out.append(ch)
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
    if in_str:
        out.append('"')
    tail = "".join(out).rstrip().rstrip(",")
    # Brackets must close in *reverse nesting order*. Counting objects and
    # arrays independently produces invalid JSON for a common truncation such
    # as {"entries":[{"work_done":"brake — it needs }]} rather than ]}}.
    closing = "".join("}" if opening == "{" else "]" for opening in reversed(stack))
    return tail + closing


def parse(text: str) -> tuple[Any | None, str]:
    """Return a JSON value and a note describing any local repair.

    Keeping recovery generic also makes the parser safe for any future
    array-shaped response.
    """
    if not text or not text.strip():
        return None, "empty_response"
    raw = text.strip()

    try:
        return json.loads(raw), ""
    except json.JSONDecodeError:
        pass

    m = _FENCE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip()), "unfenced"
        except json.JSONDecodeError:
            raw = m.group(1).strip()

    starts = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not starts:
        return None, "no_json_object"
    candidate = raw[min(starts):]

    # raw_decode accepts an otherwise-valid JSON prefix, so prose after the
    # object (a surprisingly common model habit) does not poison recovery.
    try:
        obj, _end = json.JSONDecoder().raw_decode(candidate)
        return obj, "trimmed"
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_balance(candidate)), "repaired_truncated"
    except json.JSONDecodeError as exc:
        return None, f"unparseable:{exc.msg[:40]}"


def coerce_number(v: Any) -> float | None:
    """'Rs 1,50,000.00' -> 150000.0 ; returns None rather than raising."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    s = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    if s in ("", "-", ".", "-."):
        return None
    try:
        return float(s)
    except ValueError:
        return None
