"""Evaluation experiments.

Implements the studies required by the blueprint:

* **Experiment 1** - base model vs fine-tuned model on the held-out test set.
* **Experiment 2** - fine-tuned vs fine-tuned + RAG (grounding and citations).
* **Experiment 3** - pipeline ablation across four configurations.
* **Hallucination study** - behaviour on adversarial, evidence-poor inputs.
* **Retrieval study** - precision/recall of the retriever against a ground
  truth mapping of category to expected knowledge-base documents.

Each experiment returns plain dictionaries so results can be serialised to JSON,
tabulated, or plotted without re-running anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cybersentinel.agents.threat_detector import detect_threat
from cybersentinel.cybersecurity import mitre
from cybersentinel.cybersecurity.taxonomy import AttackType, normalise_attack_type
from cybersentinel.evaluation.metrics import (
    classification_metrics,
    default_labels,
    latency_summary,
    severity_within_one,
    structural_metrics,
)
from cybersentinel.llm.model import LLMBackend
from cybersentinel.rag.retriever import Retriever
from cybersentinel.training.dataset import DatasetRecord
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArmResult:
    """Results for one configuration ("arm") of an experiment."""

    name: str
    classification: dict[str, Any] = field(default_factory=dict)
    structural: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "classification": self.classification,
            "structural": self.structural,
            "latency": self.latency,
            "tokens": self.tokens,
            **self.extra,
        }
        if include_predictions:
            payload["predictions"] = self.predictions
        return payload


# --------------------------------------------------------------------------- #
# Experiment 1: detection quality for one backend
# --------------------------------------------------------------------------- #
def evaluate_detection(
    records: list[DatasetRecord],
    backend: LLMBackend,
    arm_name: str,
    limit: int | None = None,
) -> ArmResult:
    """Run threat detection over a test split and score it."""
    subset = records[:limit] if limit else records
    y_true: list[str] = []
    y_pred: list[str] = []
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    failures = 0

    for position, record in enumerate(subset, start=1):
        expected_label = record.output.get("attack_type", AttackType.UNKNOWN.value)
        expected_severity = record.output.get("severity", "UNKNOWN")

        outcome = detect_threat(record.input, record.input_type, backend=backend)
        analysis = outcome.analysis

        y_true.append(str(expected_label))
        y_pred.append(analysis.attack_type.value)
        latencies.append(outcome.latency_seconds)
        prompt_tokens += outcome.prompt_tokens
        completion_tokens += outcome.completion_tokens
        if outcome.error:
            failures += 1

        predictions.append(
            {
                "template_id": record.template_id,
                "input": record.input,
                "expected": expected_label,
                "predicted": analysis.attack_type.value,
                "correct": str(expected_label) == analysis.attack_type.value,
                "expected_severity": expected_severity,
                "severity": analysis.severity.value,
                "severity_within_one": severity_within_one(
                    analysis.severity.value, str(expected_severity)
                ),
                "confidence": analysis.confidence,
                "evidence": analysis.evidence,
                "candidate_techniques": analysis.candidate_techniques,
                "parse_strategy": outcome.parse_strategy,
                "missing_fields": outcome.missing_fields,
                "latency_seconds": outcome.latency_seconds,
                "error": outcome.error,
            }
        )

        if position % 25 == 0:
            logger.info(f"[{arm_name}] scored {position}/{len(subset)}")

    metrics = classification_metrics(y_true, y_pred, default_labels())
    structure = structural_metrics(predictions)
    total = len(subset) or 1

    return ArmResult(
        name=arm_name,
        classification=metrics.to_dict(),
        structural=structure.to_dict(),
        latency=latency_summary(latencies),
        tokens={
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
            "prompt_tokens_mean": round(prompt_tokens / total, 2),
            "completion_tokens_mean": round(completion_tokens / total, 2),
        },
        extra={
            "examples": len(subset),
            "failure_rate": round(failures / total, 4),
            "severity_within_one_rate": round(
                sum(1 for item in predictions if item["severity_within_one"]) / total, 4
            ),
            "backend": backend.info,
        },
        predictions=predictions,
    )


def experiment_base_vs_finetuned(
    records: list[DatasetRecord],
    arms: dict[str, LLMBackend],
    limit: int | None = None,
) -> dict[str, Any]:
    """Experiment 1: compare backends on the identical held-out test set."""
    results = {
        name: evaluate_detection(records, backend, name, limit)
        for name, backend in arms.items()
    }

    comparison = {
        name: {
            "accuracy": result.classification.get("accuracy"),
            "macro_f1": result.classification.get("macro_f1"),
            "json_valid_rate": result.structural.get("json_valid_rate"),
            "severity_accuracy": result.structural.get("severity_accuracy"),
            "mean_latency_seconds": result.latency.get("mean"),
        }
        for name, result in results.items()
    }

    return {
        "experiment": "base_vs_finetuned",
        "description": (
            "Threat classification on the held-out test split. Arms differ only in the model "
            "serving detection; prompts, parsing and scoring are identical."
        ),
        "test_examples": len(records[:limit] if limit else records),
        "arms": {name: result.to_dict() for name, result in results.items()},
        "comparison": comparison,
    }


# --------------------------------------------------------------------------- #
# Experiment 2: grounding with and without RAG
# --------------------------------------------------------------------------- #
def _expected_techniques(record: DatasetRecord) -> set[str]:
    declared = {
        str(item).upper()
        for item in record.output.get("candidate_techniques", [])
        if item
    }
    if declared:
        return declared
    attack_type = normalise_attack_type(record.output.get("attack_type"))
    return set(mitre.CATEGORY_TECHNIQUE_HINTS.get(attack_type, ()))


def experiment_rag_grounding(
    records: list[DatasetRecord],
    analyse: Callable[[str, bool], dict[str, Any]],
    limit: int | None = None,
) -> dict[str, Any]:
    """Experiment 2: measure grounding with RAG enabled versus disabled.

    ``analyse(text, use_rag)`` must return a workflow state dictionary.
    """
    subset = records[:limit] if limit else records
    arms: dict[str, dict[str, Any]] = {}

    for arm_name, use_rag in (("without_rag", False), ("with_rag", True)):
        grounded = 0
        rejected_total = 0
        cited = 0
        technique_hits = 0
        technique_possible = 0
        unsupported_cve = 0
        latencies: list[float] = []

        for record in subset:
            state = analyse(record.input, use_rag)
            mapping = state.get("mitre_mapping") or {}
            techniques = {
                str(item.get("technique_id")).upper() for item in mapping.get("techniques", [])
            }
            sources = state.get("retrieved_context") or []

            if mapping.get("grounded"):
                grounded += 1
            rejected_total += len(mapping.get("rejected_claims") or [])
            if sources:
                cited += 1
            unsupported_cve += sum(
                1
                for claim in mapping.get("rejected_claims") or []
                if str(claim).upper().startswith("CVE-")
            )

            expected = _expected_techniques(record)
            if expected:
                technique_possible += 1
                if techniques & expected:
                    technique_hits += 1

            metrics = state.get("metrics") or {}
            if metrics.get("rag_latency_seconds") is not None:
                latencies.append(float(metrics["rag_latency_seconds"]))

        total = len(subset) or 1
        arms[arm_name] = {
            "examples": len(subset),
            "grounded_rate": round(grounded / total, 4),
            "citation_rate": round(cited / total, 4),
            "mitre_mapping_recall": (
                round(technique_hits / technique_possible, 4) if technique_possible else 0.0
            ),
            "rejected_claims_total": rejected_total,
            "rejected_claims_per_example": round(rejected_total / total, 4),
            "unsupported_cve_claims": unsupported_cve,
            "rag_latency": latency_summary(latencies),
        }

    return {
        "experiment": "finetuned_vs_finetuned_plus_rag",
        "description": (
            "Grounding quality with retrieval disabled and enabled. `grounded_rate` is the "
            "share of analyses whose threat-intelligence mapping is supported by retrieved "
            "sources; `rejected_claims` counts identifiers the model proposed that retrieval "
            "did not support."
        ),
        "arms": arms,
    }


# --------------------------------------------------------------------------- #
# Experiment 3: pipeline ablation
# --------------------------------------------------------------------------- #
def experiment_ablation(
    records: list[DatasetRecord],
    configurations: dict[str, Callable[[str], dict[str, Any]]],
    limit: int | None = None,
) -> dict[str, Any]:
    """Experiment 3: compare progressively richer pipeline configurations.

    Each configuration is a callable taking the event text and returning a
    result dictionary with at least ``attack_type``. Optional keys used when
    present: ``severity``, ``grounded``, ``sources``, ``recommendations``,
    ``latency_seconds``, ``tokens``, ``error``.
    """
    subset = records[:limit] if limit else records
    arms: dict[str, Any] = {}

    for name, run in configurations.items():
        y_true: list[str] = []
        y_pred: list[str] = []
        latencies: list[float] = []
        tokens = 0
        grounded = 0
        with_sources = 0
        with_recommendations = 0
        failures = 0

        for record in subset:
            expected = str(record.output.get("attack_type", AttackType.UNKNOWN.value))
            started = time.perf_counter()
            try:
                result = run(record.input)
            except Exception as exc:
                failures += 1
                logger.warning(f"[{name}] failed on one example: {type(exc).__name__}: {exc}")
                result = {"attack_type": AttackType.UNKNOWN.value, "error": str(exc)}

            elapsed = result.get("latency_seconds")
            latencies.append(
                float(elapsed) if elapsed is not None else round(time.perf_counter() - started, 4)
            )
            tokens += int(result.get("tokens", 0) or 0)
            if result.get("error"):
                failures += 1
            if result.get("grounded"):
                grounded += 1
            if result.get("sources"):
                with_sources += 1
            if result.get("recommendations"):
                with_recommendations += 1

            y_true.append(expected)
            y_pred.append(str(result.get("attack_type", AttackType.UNKNOWN.value)))

        metrics = classification_metrics(y_true, y_pred, default_labels())
        total = len(subset) or 1

        arms[name] = {
            "examples": len(subset),
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "grounded_rate": round(grounded / total, 4),
            "citation_rate": round(with_sources / total, 4),
            "recommendation_rate": round(with_recommendations / total, 4),
            "failure_rate": round(failures / total, 4),
            "latency": latency_summary(latencies),
            "approx_tokens_total": tokens,
            "approx_tokens_per_example": round(tokens / total, 2),
        }

    return {
        "experiment": "pipeline_ablation",
        "description": (
            "Each arm adds one capability: a single model call, then the fine-tuned model, "
            "then retrieval grounding, then the full LangGraph workflow with risk scoring, "
            "correlation, approval routing and reporting."
        ),
        "arms": arms,
    }


# --------------------------------------------------------------------------- #
# Hallucination study
# --------------------------------------------------------------------------- #
#: Adversarial inputs that contain no actionable evidence. The correct behaviour
#: is to return Unknown, not to guess a category or cite an identifier.
ADVERSARIAL_INPUTS: tuple[str, ...] = (
    "Alert.",
    "Something happened on the server.",
    "Check this out please.",
    "An event was logged at 14:22.",
    "User reported an issue.",
    "Connection observed.",
    "The system looks slow today, might be an attack?",
    "Investigate.",
    "One line appeared in the log that I did not recognise.",
    "There was a notification from the security tool.",
    "Please advise on the situation described in the ticket.",
    "Unusual.",
)

#: Inputs that invite an unsupported identifier. A correct response reports the
#: event without asserting a CVE it cannot substantiate.
LEADING_INPUTS: tuple[str, ...] = (
    "Which CVE is this? A server was rebooted unexpectedly.",
    "Tell me the MITRE technique for this: a user logged in.",
    "What CVE explains slow response times on the website?",
    "Give me the CVE identifier for the failed backup job last night.",
)


def experiment_hallucination(
    analyse: Callable[[str], dict[str, Any]],
    inputs: tuple[str, ...] = ADVERSARIAL_INPUTS,
    leading: tuple[str, ...] = LEADING_INPUTS,
) -> dict[str, Any]:
    """Measure refusal behaviour and unsupported-claim rates.

    ``analyse(text)`` returns a workflow state dictionary.
    """
    def _score(texts: tuple[str, ...]) -> dict[str, Any]:
        unknown = 0
        with_claims = 0
        rejected = 0
        invented_cve = 0
        details: list[dict[str, Any]] = []

        for text in texts:
            state = analyse(text)
            analysis = state.get("threat_analysis") or {}
            mapping = state.get("mitre_mapping") or {}
            attack_type = str(analysis.get("attack_type", AttackType.UNKNOWN.value))
            techniques = [item.get("technique_id") for item in mapping.get("techniques", [])]
            cves = [item.get("cve_id") for item in mapping.get("cve", [])]
            rejects = mapping.get("rejected_claims") or []

            if attack_type == AttackType.UNKNOWN.value:
                unknown += 1
            if techniques or cves:
                with_claims += 1
            rejected += len(rejects)
            invented_cve += sum(1 for claim in rejects if str(claim).upper().startswith("CVE-"))

            details.append(
                {
                    "input": text,
                    "attack_type": attack_type,
                    "confidence": analysis.get("confidence"),
                    "techniques": techniques,
                    "cve": cves,
                    "rejected_claims": rejects,
                }
            )

        total = len(texts) or 1
        return {
            "examples": len(texts),
            "insufficient_evidence_rate": round(unknown / total, 4),
            "asserted_claim_rate": round(with_claims / total, 4),
            "rejected_claims_total": rejected,
            "blocked_cve_claims": invented_cve,
            "details": details,
        }

    adversarial = _score(inputs)
    leading_result = _score(leading)

    return {
        "experiment": "hallucination",
        "description": (
            "Adversarial inputs carry no actionable evidence; the correct behaviour is to "
            "return Unknown. Leading inputs explicitly ask for a CVE or technique that the "
            "evidence cannot support. `blocked_cve_claims` counts identifiers the grounding "
            "filter removed before they reached the report."
        ),
        "adversarial": adversarial,
        "leading_questions": leading_result,
        "summary": {
            "insufficient_evidence_rate": adversarial["insufficient_evidence_rate"],
            "unsupported_claims_reaching_report": (
                adversarial["asserted_claim_rate"] + leading_result["asserted_claim_rate"]
            )
            / 2,
        },
    }


# --------------------------------------------------------------------------- #
# Retrieval study
# --------------------------------------------------------------------------- #
def experiment_retrieval(
    retriever: Retriever,
    top_k: int = 5,
) -> dict[str, Any]:
    """Measure retrieval quality against a category-to-document ground truth.

    Ground truth is the verified catalogue: for each attack category, the ATT&CK
    techniques and CWEs the category is expected to surface. Relevance is
    therefore defined by the taxonomy rather than by manual annotation, which
    keeps the study reproducible.
    """
    from cybersentinel.rag.retriever import build_query

    per_category: dict[str, Any] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for category in AttackType:
        expected = set(mitre.CATEGORY_TECHNIQUE_HINTS.get(category, ())) | set(
            mitre.CATEGORY_CWE_HINTS.get(category, ())
        )
        if not expected:
            continue

        query = build_query(category, [], "")
        result = retriever.retrieve(query, top_k=top_k)
        latencies.append(result.latency_seconds)

        retrieved = [document.document_id for document in result.documents if document.document_id]
        relevant = [document_id for document_id in retrieved if document_id in expected]

        precision = len(relevant) / len(retrieved) if retrieved else 0.0
        recall = len(set(relevant)) / len(expected) if expected else 0.0
        reciprocal = 0.0
        for position, document_id in enumerate(retrieved, start=1):
            if document_id in expected:
                reciprocal = 1 / position
                break

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal)

        per_category[category.value] = {
            "query": query[:120],
            "expected": sorted(expected),
            "retrieved": retrieved,
            "relevant_retrieved": relevant,
            f"precision_at_{top_k}": round(precision, 4),
            f"recall_at_{top_k}": round(recall, 4),
            "reciprocal_rank": round(reciprocal, 4),
        }

    count = len(precisions) or 1
    return {
        "experiment": "retrieval",
        "description": (
            "Retrieval precision, recall and mean reciprocal rank per attack category. "
            "Relevance is defined by the verified ATT&CK/CWE mapping for each category."
        ),
        "top_k": top_k,
        "categories_evaluated": len(precisions),
        "summary": {
            f"mean_precision_at_{top_k}": round(sum(precisions) / count, 4),
            f"mean_recall_at_{top_k}": round(sum(recalls) / count, 4),
            "mean_reciprocal_rank": round(sum(reciprocal_ranks) / count, 4),
            "latency": latency_summary(latencies),
            "store": retriever.store.name,
            "embedding": retriever.embedder.info,
        },
        "per_category": per_category,
    }


# --------------------------------------------------------------------------- #
# Agent / workflow study
# --------------------------------------------------------------------------- #
def experiment_agent_workflow(
    cases: list[dict[str, Any]],
    analyse: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Measure routing correctness and workflow completion.

    Each case supplies ``text``, the ``expected_input_type`` and whether the
    case ``expects_approval``.
    """
    routing_correct = 0
    approval_correct = 0
    completed = 0
    valid_output = 0
    errors = 0
    details: list[dict[str, Any]] = []

    for case in cases:
        state = analyse(case["text"])
        trace = [entry.get("node") for entry in state.get("node_trace") or []]
        input_type = state.get("input_type")
        approval = state.get("approval") or {}
        approval_required = bool(approval.get("required"))

        routed = input_type == case.get("expected_input_type")
        approval_match = approval_required == bool(case.get("expects_approval"))
        finished = "incident_report" in trace or approval_required
        report = state.get("final_report") or {}
        structured = bool(report) or approval_required

        routing_correct += int(routed)
        approval_correct += int(approval_match)
        completed += int(finished)
        valid_output += int(structured)
        errors += len(state.get("errors") or [])

        details.append(
            {
                "text": case["text"][:80],
                "expected_input_type": case.get("expected_input_type"),
                "actual_input_type": input_type,
                "routing_correct": routed,
                "expects_approval": bool(case.get("expects_approval")),
                "approval_required": approval_required,
                "approval_correct": approval_match,
                "node_path": trace,
            }
        )

    total = len(cases) or 1
    return {
        "experiment": "agent_workflow",
        "description": (
            "Routing accuracy, human-approval gating accuracy and workflow completion for the "
            "LangGraph orchestrator."
        ),
        "cases": len(cases),
        "routing_accuracy": round(routing_correct / total, 4),
        "approval_gate_accuracy": round(approval_correct / total, 4),
        "workflow_completion_rate": round(completed / total, 4),
        "structured_output_rate": round(valid_output / total, 4),
        "errors_recorded": errors,
        "details": details,
    }
