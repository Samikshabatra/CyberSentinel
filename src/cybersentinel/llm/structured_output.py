"""Robust parsing of model output into validated Pydantic objects.

LLMs emit malformed JSON. The recovery ladder implemented here is:

1. Parse the raw text as JSON.
2. Strip markdown fences and surrounding prose, then parse the first balanced
   JSON object found.
3. Apply conservative textual repairs (trailing commas, single quotes,
   Python literals, unquoted keys) and parse again.
4. Give up and return a typed failure.

Values are never fabricated to satisfy the schema. When a required field is
missing the safe default (`Unknown` / `0.0`) is used and the omission is
recorded so evaluation can measure field completeness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cybersentinel.cybersecurity.taxonomy import (
    normalise_attack_type,
    normalise_severity,
)
from cybersentinel.schemas.analysis import ThreatAnalysis

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")


@dataclass
class ParseResult:
    """Outcome of parsing one model response."""

    data: dict[str, Any] | None
    valid_json: bool
    strategy: str
    missing_fields: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.data is not None


def _find_balanced_object(text: str) -> str | None:
    """Return the first balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair(text: str) -> str:
    """Conservative textual fixes for common LLM JSON mistakes."""
    repaired = _TRAILING_COMMA.sub(r"\1", text)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = _UNQUOTED_KEY.sub(r'\1"\2"\3', repaired)
    # Single-quoted strings -> double-quoted, only when no double quotes are used.
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def parse_json_object(raw: str) -> ParseResult:
    """Extract a JSON object from raw model output using the recovery ladder."""
    if not raw or not raw.strip():
        return ParseResult(None, False, "empty", error="model returned empty output")

    text = raw.strip()

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return ParseResult(loaded, True, "direct")
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_PATTERN.search(text)
    candidate = fenced.group(1).strip() if fenced else text

    block = _find_balanced_object(candidate)
    if block is not None:
        try:
            loaded = json.loads(block)
            if isinstance(loaded, dict):
                return ParseResult(loaded, True, "extracted")
        except json.JSONDecodeError:
            try:
                loaded = json.loads(_repair(block))
                if isinstance(loaded, dict):
                    return ParseResult(loaded, False, "repaired")
            except json.JSONDecodeError as exc:
                return ParseResult(None, False, "failed", error=f"invalid JSON: {exc.msg}")

    return ParseResult(None, False, "failed", error="no JSON object found in model output")


def _coerce_confidence(value: Any) -> float:
    """Coerce a confidence value into [0, 1]; percentages are rescaled."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0:
        confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
    return max(0.0, min(1.0, confidence))


def _coerce_str_list(value: Any) -> list[str]:
    """Accept a list, a delimited string, or a single value."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[;\n]|(?<=[a-z0-9])\s*,\s*(?=[A-Z])", value)
        return [part.strip(" -*\t") for part in parts if part.strip(" -*\t")]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("description") or item.get("text") or item.get("evidence")
                if isinstance(text, str) and text.strip():
                    items.append(text.strip())
            elif item is not None:
                items.append(str(item))
        return items
    return [str(value)]


def parse_threat_analysis(raw: str, model_source: str = "unknown") -> tuple[ThreatAnalysis, ParseResult]:
    """Parse raw model output into a validated `ThreatAnalysis`.

    On failure a conservative `Unknown` analysis is returned - never a guess.
    """
    result = parse_json_object(raw)

    if not result.ok:
        return (
            ThreatAnalysis(
                attack_type=normalise_attack_type(None),
                reasoning="Model output could not be parsed as JSON; no classification asserted.",
                model_source=model_source,
            ),
            result,
        )

    data = result.data or {}
    required = ("attack_type", "severity", "confidence", "evidence")
    result.missing_fields = [key for key in required if key not in data]

    # Accept a few common key aliases before falling back to defaults.
    attack_raw = data.get("attack_type") or data.get("attack") or data.get("classification")
    severity_raw = data.get("severity") or data.get("risk") or data.get("priority")
    techniques = _coerce_str_list(
        data.get("candidate_techniques")
        or data.get("mitre_techniques")
        or data.get("mitre_technique")
        or data.get("techniques")
    )

    analysis = ThreatAnalysis(
        attack_type=normalise_attack_type(attack_raw if isinstance(attack_raw, str) else None),
        severity=normalise_severity(severity_raw if isinstance(severity_raw, str) else None),
        confidence=_coerce_confidence(data.get("confidence")),
        evidence=_coerce_str_list(data.get("evidence") or data.get("indicators")),
        candidate_techniques=[technique.upper() for technique in techniques],
        reasoning=str(data.get("reasoning") or data.get("explanation") or "").strip(),
        model_source=model_source,
    )
    return analysis, result


def json_validity_report(raw_outputs: list[str]) -> dict[str, float]:
    """Aggregate JSON-validity statistics over a batch (used by evaluation)."""
    if not raw_outputs:
        return {"total": 0, "valid_rate": 0.0, "repaired_rate": 0.0, "failed_rate": 0.0}

    counts = {"direct": 0, "extracted": 0, "repaired": 0, "failed": 0, "empty": 0}
    for raw in raw_outputs:
        counts[parse_json_object(raw).strategy] += 1

    total = len(raw_outputs)
    strict_valid = counts["direct"] + counts["extracted"]
    return {
        "total": total,
        "valid_rate": round(strict_valid / total, 4),
        "repaired_rate": round(counts["repaired"] / total, 4),
        "failed_rate": round((counts["failed"] + counts["empty"]) / total, 4),
    }
