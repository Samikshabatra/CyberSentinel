"""High-level inference helpers used by the agents.

Wraps a backend with timing, a single safe retry, token accounting and typed
failure handling. Agents never call a backend directly - they call these
helpers so that latency and failures are measured consistently for the
performance evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cybersentinel.llm.model import LLMBackend, LLMError, get_backend
from cybersentinel.llm.prompts import build_detection_messages
from cybersentinel.llm.structured_output import ParseResult, parse_threat_analysis
from cybersentinel.schemas.analysis import ThreatAnalysis
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """One backend call: text, latency, rough token counts and any error."""

    text: str
    latency_seconds: float
    backend: str
    prompt_chars: int = 0
    completion_chars: int = 0
    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def approx_prompt_tokens(self) -> int:
        """Rough token estimate (~4 characters per token) for cost reporting."""
        return round(self.prompt_chars / 4)

    @property
    def approx_completion_tokens(self) -> int:
        return round(self.completion_chars / 4)


def generate(
    messages: list[dict[str, str]],
    backend: LLMBackend | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 1,
) -> GenerationResult:
    """Call the backend with timing and one retry on transient failure.

    Retries are safe here: generation has no side effects. A retry uses
    temperature 0 to make the second attempt more likely to be well-formed.
    """
    llm = backend or get_backend()
    prompt_chars = sum(len(message["content"]) for message in messages)
    started = time.perf_counter()
    last_error: str | None = None

    for attempt in range(1, retries + 2):
        try:
            text = llm.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=0.0 if attempt > 1 else temperature,
            )
            if not text.strip():
                raise LLMError("backend returned empty output")
            return GenerationResult(
                text=text,
                latency_seconds=round(time.perf_counter() - started, 3),
                backend=llm.name,
                prompt_chars=prompt_chars,
                completion_chars=len(text),
                attempts=attempt,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"generation attempt {attempt} failed: {last_error}")

    return GenerationResult(
        text="",
        latency_seconds=round(time.perf_counter() - started, 3),
        backend=llm.name,
        prompt_chars=prompt_chars,
        attempts=retries + 1,
        error=last_error,
    )


def analyze_event(
    event_text: str,
    input_type: str = "alert",
    backend: LLMBackend | None = None,
    model_source: str | None = None,
) -> tuple[ThreatAnalysis, GenerationResult, ParseResult]:
    """Run threat detection on one event and return a validated analysis."""
    llm = backend or get_backend()
    source = model_source or _describe_source(llm)

    messages = build_detection_messages(event_text, input_type)
    generation = generate(messages, backend=llm)

    if not generation.ok:
        analysis = ThreatAnalysis(
            reasoning=f"Detection unavailable: {generation.error}",
            model_source=source,
        )
        return analysis, generation, ParseResult(None, False, "skipped", error=generation.error)

    analysis, parse_result = parse_threat_analysis(generation.text, model_source=source)
    return analysis, generation, parse_result


def _describe_source(backend: LLMBackend) -> str:
    """Label the analysis with what actually produced it (for the ablation)."""
    info = backend.info
    if info.get("backend") == "mock":
        return "mock"
    return "finetuned" if info.get("adapter") else "base"
