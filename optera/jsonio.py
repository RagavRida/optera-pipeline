"""Tolerant JSON recovery from model output.

Models occasionally wrap JSON in a markdown fence, prepend a sentence, or get
truncated by max_tokens mid-object. Re-calling the model to fix formatting costs
real money, so we repair locally first and only escalate if repair fails.
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
    out, depth_obj, depth_arr, in_str, esc = [], 0, 0, False, False
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
            if ch == "{":
                depth_obj += 1
            elif ch == "}":
                depth_obj -= 1
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                depth_arr -= 1
    if in_str:
        out.append('"')
    tail = "".join(out).rstrip().rstrip(",")
    return tail + "]" * max(depth_arr, 0) + "}" * max(depth_obj, 0)


def parse(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return (object, note). note describes any repair that was applied."""
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

    start = raw.find("{")
    if start == -1:
        return None, "no_json_object"
    candidate = raw[start:]

    end = candidate.rfind("}")
    if end != -1:
        try:
            return json.loads(candidate[: end + 1]), "trimmed"
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
