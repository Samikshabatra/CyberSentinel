"""Evaluation runner: wires the experiments to real components.

Defines the ablation arms in one place so the experiment definitions stay
independent of how a given arm happens to be built.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cybersentinel.agents.threat_detector import detect_threat
from cybersentinel.evaluation import experiments
from cybersentinel.graph.workflow import CyberSentinelWorkflow
from cybersentinel.llm.model import LLMBackend, MockBackend, build_backend
from cybersentinel.rag.retriever import Retriever
from cybersentinel.rag.vectorstore import build_vector_store
from cybersentinel.training.dataset import DatasetRecord, load_records
from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

#: Routing cases for the agent-workflow study.
ROUTING_CASES: list[dict[str, Any]] = [
    {
        "text": "47 failed SSH login attempts from 198.51.100.23 within 3 minutes for user root.",
        "expected_input_type": "alert",
        "expects_approval": True,
    },
    {
        "text": (
            "From: security@example.com\nSubject: Urgent - verify your account\n\n"
            "Confirm your credentials at http://verify.example.net/login or your mailbox "
            "will be suspended."
        ),
        "expected_input_type": "email",
        "expects_approval": True,
    },
    {
        "text": "http://login-secure.example.net/account/verify?id=8837",
        "expected_input_type": "url",
        "expects_approval": False,
    },
    {
        "text": (
            "2026-03-04T08:12:31Z web-prod-01 sshd[2211]: Failed password for admin from "
            "198.51.100.23\n"
            "2026-03-04T08:12:34Z web-prod-01 sshd[2212]: Failed password for admin from "
            "198.51.100.23\n"
            "2026-03-04T08:12:37Z web-prod-01 sshd[2213]: Failed password for admin from "
            "198.51.100.23"
        ),
        "expected_input_type": "log",
        "expects_approval": True,
    },
    {
        "text": (
            "Vulnerability scan reports db-prod-02 running an unpatched application server "
            "with a published remote code execution advisory and a CVSS score of 9.8."
        ),
        "expected_input_type": "vulnerability",
        "expects_approval": True,
    },
    {
        "text": (
            "Event 1: Port scan from 203.0.113.45 against 1200 ports\n\n"
            "Event 2: 20 failed SSH logins for user admin from 203.0.113.45\n\n"
            "Event 3: Successful SSH login for user admin from 203.0.113.45\n\n"
            "Event 4: User admin added to the administrators group shortly after login"
        ),
        "expected_input_type": "multi_event",
        "expects_approval": True,
    },
    {
        "text": (
            "User alice authenticated successfully to the VPN at 08:52 from the usual office "
            "address, matching their normal weekday pattern. Routine activity."
        ),
        "expected_input_type": "alert",
        "expects_approval": False,
    },
]


@dataclass
class EvaluationContext:
    """Components shared by every experiment in a run."""

    settings: Settings
    retriever: Retriever
    workflow: CyberSentinelWorkflow
    backends: dict[str, LLMBackend]

    def analyse_state(self, text: str, use_rag: bool = True) -> dict[str, Any]:
        """Run the full workflow and return the final state."""
        run = self.workflow.analyze(text, use_rag=use_rag, use_llm_response=True)
        return run.state


def build_context(
    adapter_path: str | None = None,
    prefer_local_store: bool = True,
    include_hf_arms: bool = False,
) -> EvaluationContext:
    """Assemble evaluation components.

    The mock backend is always present: it is the reproducible ``rules``
    baseline and keeps the whole harness runnable without a GPU. Hugging Face
    arms are added only when explicitly requested, because they need the ML
    extra and a downloaded model.
    """
    settings = get_settings()
    backends: dict[str, LLMBackend] = {"rules_baseline": MockBackend()}

    if include_hf_arms:
        try:
            base_settings = settings.model_copy(update={"model_adapter_path": None})
            backends["base_model"] = build_backend(base_settings, backend="hf")
            if adapter_path:
                tuned_settings = settings.model_copy(update={"model_adapter_path": adapter_path})
                backends["finetuned_model"] = build_backend(tuned_settings, backend="hf")
            else:
                logger.warning("no adapter path supplied: the fine-tuned arm is skipped")
        except Exception as exc:
            logger.warning(
                f"Hugging Face arms unavailable ({type(exc).__name__}: {exc}); "
                "continuing with the rules baseline only"
            )

    retriever = Retriever(store=build_vector_store(settings, prefer_local=prefer_local_store))
    workflow = CyberSentinelWorkflow(
        backend=backends.get("finetuned_model") or backends["rules_baseline"],
        retriever=retriever,
        enable_interrupt=False,  # evaluation runs the pipeline end to end
    )

    return EvaluationContext(
        settings=settings, retriever=retriever, workflow=workflow, backends=backends
    )


def _single_call_arm(backend: LLMBackend, retriever: Retriever | None = None):
    """Arm 1/2: one model call, no orchestration."""

    def run(text: str) -> dict[str, Any]:
        outcome = detect_threat(text, backend=backend)
        return {
            "attack_type": outcome.analysis.attack_type.value,
            "severity": outcome.analysis.severity.value,
            "grounded": False,
            "sources": [],
            "recommendations": [],
            "latency_seconds": outcome.latency_seconds,
            "tokens": outcome.prompt_tokens + outcome.completion_tokens,
            "error": outcome.error,
        }

    return run


def _rag_arm(backend: LLMBackend, retriever: Retriever):
    """Arm 3: model plus retrieval grounding, still no orchestration."""
    from cybersentinel.agents.threat_intelligence import gather_intelligence

    def run(text: str) -> dict[str, Any]:
        started = time.perf_counter()
        outcome = detect_threat(text, backend=backend)
        intel = gather_intelligence(
            outcome.analysis, text, retriever=retriever, backend=backend
        )
        return {
            "attack_type": outcome.analysis.attack_type.value,
            "severity": outcome.analysis.severity.value,
            "grounded": intel.mapping.grounded,
            "sources": intel.sources,
            "recommendations": [],
            "latency_seconds": round(time.perf_counter() - started, 4),
            "tokens": outcome.prompt_tokens + outcome.completion_tokens,
            "error": outcome.error or intel.error,
        }

    return run


def _workflow_arm(context: EvaluationContext):
    """Arm 4: the full LangGraph workflow."""

    def run(text: str) -> dict[str, Any]:
        started = time.perf_counter()
        state = context.analyse_state(text, use_rag=True)
        analysis = state.get("threat_analysis") or {}
        mapping = state.get("mitre_mapping") or {}
        metrics = state.get("metrics") or {}
        return {
            "attack_type": analysis.get("attack_type"),
            "severity": (state.get("risk_assessment") or {}).get("risk_level"),
            "grounded": bool(mapping.get("grounded")),
            "sources": state.get("retrieved_context") or [],
            "recommendations": state.get("response_recommendations") or [],
            "latency_seconds": round(time.perf_counter() - started, 4),
            "tokens": int(metrics.get("detection_prompt_tokens", 0))
            + int(metrics.get("detection_completion_tokens", 0)),
            "error": "; ".join(state.get("errors") or []) or None,
        }

    return run


def run_all(
    context: EvaluationContext,
    test_records: list[DatasetRecord],
    limit: int | None = None,
    output_dir: Path | None = None,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """Run every experiment and return the combined results."""
    started = time.perf_counter()
    detection_backend = context.backends.get("finetuned_model") or context.backends["rules_baseline"]

    logger.info("running experiment 1: base vs fine-tuned (template test set)")
    experiment_1 = experiments.experiment_base_vs_finetuned(test_records, context.backends, limit)

    logger.info("running experiment 1b: hard test set")
    hard_records = load_hard_test_records(context.settings)
    experiment_1b = (
        experiments.experiment_base_vs_finetuned(hard_records, context.backends, None)
        if hard_records
        else {"experiment": "hard_test", "note": "hard test set not found"}
    )

    logger.info("running experiment 2: RAG grounding")
    experiment_2 = experiments.experiment_rag_grounding(
        test_records,
        lambda text, use_rag: context.analyse_state(text, use_rag=use_rag),
        limit=min(limit or 40, 40),
    )

    logger.info("running experiment 3: pipeline ablation")
    ablation_arms = {
        "1_single_call_rules": _single_call_arm(context.backends["rules_baseline"]),
        "3_detection_plus_rag": _rag_arm(detection_backend, context.retriever),
        "4_full_langgraph_workflow": _workflow_arm(context),
    }
    if "base_model" in context.backends:
        ablation_arms["0_base_model_single_call"] = _single_call_arm(context.backends["base_model"])
    if "finetuned_model" in context.backends:
        ablation_arms["2_finetuned_single_call"] = _single_call_arm(
            context.backends["finetuned_model"]
        )
    experiment_3 = experiments.experiment_ablation(
        test_records, dict(sorted(ablation_arms.items())), limit=min(limit or 40, 40)
    )

    logger.info("running hallucination study")
    hallucination = experiments.experiment_hallucination(
        lambda text: context.analyse_state(text, use_rag=True)
    )

    logger.info("running retrieval study")
    retrieval = experiments.experiment_retrieval(context.retriever, top_k=context.settings.top_k)

    logger.info("running agent workflow study")
    agent = experiments.experiment_agent_workflow(
        ROUTING_CASES, lambda text: context.analyse_state(text, use_rag=True)
    )

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "environment": {
            "llm_backend": context.settings.llm_backend,
            "base_model": context.settings.base_model_name,
            "adapter": context.settings.model_adapter_path,
            "embedding_backend": context.settings.embedding_backend,
            "vector_store": context.retriever.store.name,
            "arms_evaluated": sorted(context.backends),
        },
        "test_set": {
            "examples": len(test_records[:limit] if limit else test_records),
            "total_available": len(test_records),
        },
        "experiment_1_base_vs_finetuned": _strip_predictions(experiment_1, include_predictions),
        "experiment_1b_hard_test": _strip_predictions(experiment_1b, include_predictions),
        "experiment_2_rag_grounding": experiment_2,
        "experiment_3_ablation": experiment_3,
        "hallucination": hallucination,
        "retrieval": retrieval,
        "agent_workflow": agent,
    }

    if output_dir:
        write_results(results, output_dir)

    return results


def _strip_predictions(experiment: dict[str, Any], include: bool) -> dict[str, Any]:
    if include:
        return experiment
    for arm in experiment.get("arms", {}).values():
        arm.pop("predictions", None)
    return experiment


def write_results(results: dict[str, Any], output_dir: Path) -> Path:
    """Write results JSON and a readable Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    json_path = output_dir / f"evaluation-{stamp}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    latest = output_dir / "latest.json"
    latest.write_text(json.dumps(results, indent=2), encoding="utf-8")

    (output_dir / f"summary-{stamp}.md").write_text(summarise(results), encoding="utf-8")
    (output_dir / "latest_summary.md").write_text(summarise(results), encoding="utf-8")

    logger.info(f"results written to {json_path}")
    return json_path


def summarise(results: dict[str, Any]) -> str:
    """Render a Markdown summary of an evaluation run."""
    lines: list[str] = ["# CyberSentinel evaluation summary", ""]
    environment = results.get("environment", {})
    lines += [
        f"- Generated: {results.get('generated_at')}",
        f"- Runtime: {results.get('runtime_seconds')} s",
        f"- LLM backend: `{environment.get('llm_backend')}`",
        f"- Base model: `{environment.get('base_model')}`",
        f"- Adapter: `{environment.get('adapter') or 'none'}`",
        f"- Embeddings: `{environment.get('embedding_backend')}`, vector store: "
        f"`{environment.get('vector_store')}`",
        f"- Test examples: {results.get('test_set', {}).get('examples')}",
        "",
        "## Experiment 1 - detection quality",
        "",
        "| Arm | Accuracy | Macro F1 | JSON valid | Severity acc. | Mean latency (s) |",
        "|---|---|---|---|---|---|",
    ]

    for name, row in results.get("experiment_1_base_vs_finetuned", {}).get("comparison", {}).items():
        lines.append(
            f"| {name} | {row.get('accuracy')} | {row.get('macro_f1')} | "
            f"{row.get('json_valid_rate')} | {row.get('severity_accuracy')} | "
            f"{row.get('mean_latency_seconds')} |"
        )

    hard = results.get("experiment_1b_hard_test", {}).get("comparison")
    if hard:
        lines += [
            "",
            "## Experiment 1b - hard test set (paraphrases, keyword decoys, insufficient evidence)",
            "",
            "| Arm | Accuracy | Macro F1 | JSON valid | Severity acc. |",
            "|---|---|---|---|---|",
        ]
        for name, row in hard.items():
            lines.append(
                f"| {name} | {row.get('accuracy')} | {row.get('macro_f1')} | "
                f"{row.get('json_valid_rate')} | {row.get('severity_accuracy')} |"
            )
        lines += [
            "",
            "The template test set is generated from the same grammar as the training data, so a "
            "keyword baseline can saturate it. The hard set removes signature vocabulary and adds "
            "benign events containing alarming keywords; the gap between the two tables is the "
            "quantity of interest.",
        ]

    lines += [
        "",
        "## Experiment 2 - retrieval grounding",
        "",
        "| Arm | Grounded | Citations | MITRE recall | Rejected claims / example |",
        "|---|---|---|---|---|",
    ]
    for name, row in results.get("experiment_2_rag_grounding", {}).get("arms", {}).items():
        lines.append(
            f"| {name} | {row.get('grounded_rate')} | {row.get('citation_rate')} | "
            f"{row.get('mitre_mapping_recall')} | {row.get('rejected_claims_per_example')} |"
        )

    lines += [
        "",
        "## Experiment 3 - pipeline ablation",
        "",
        "| Arm | Accuracy | Macro F1 | Grounded | Recommendations | Failure rate | Mean latency (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in results.get("experiment_3_ablation", {}).get("arms", {}).items():
        lines.append(
            f"| {name} | {row.get('accuracy')} | {row.get('macro_f1')} | "
            f"{row.get('grounded_rate')} | {row.get('recommendation_rate')} | "
            f"{row.get('failure_rate')} | {row.get('latency', {}).get('mean')} |"
        )

    hallucination = results.get("hallucination", {})
    adversarial = hallucination.get("adversarial", {})
    leading = hallucination.get("leading_questions", {})
    lines += [
        "",
        "## Hallucination and grounding",
        "",
        f"- Insufficient-evidence rate on adversarial inputs: "
        f"**{adversarial.get('insufficient_evidence_rate')}** (higher is better)",
        f"- Claims asserted on adversarial inputs: {adversarial.get('asserted_claim_rate')} "
        "(lower is better)",
        f"- Claims asserted on leading questions: {leading.get('asserted_claim_rate')}",
        f"- CVE claims blocked by the grounding filter: "
        f"{adversarial.get('blocked_cve_claims', 0) + leading.get('blocked_cve_claims', 0)}",
    ]

    retrieval = results.get("retrieval", {}).get("summary", {})
    top_k = results.get("retrieval", {}).get("top_k", 5)
    lines += [
        "",
        "## Retrieval",
        "",
        f"- Mean precision@{top_k}: {retrieval.get(f'mean_precision_at_{top_k}')}",
        f"- Mean recall@{top_k}: {retrieval.get(f'mean_recall_at_{top_k}')}",
        f"- Mean reciprocal rank: {retrieval.get('mean_reciprocal_rank')}",
    ]

    agent = results.get("agent_workflow", {})
    lines += [
        "",
        "## Agent workflow",
        "",
        f"- Routing accuracy: {agent.get('routing_accuracy')}",
        f"- Approval-gate accuracy: {agent.get('approval_gate_accuracy')}",
        f"- Workflow completion: {agent.get('workflow_completion_rate')}",
        f"- Structured output rate: {agent.get('structured_output_rate')}",
        "",
        "---",
        "",
        "Results are produced by `scripts/evaluate.py`. Figures describe this system on this "
        "test set only; no claim of general cybersecurity accuracy is made.",
    ]

    return "\n".join(lines) + "\n"


def load_test_records(settings: Settings | None = None) -> list[DatasetRecord]:
    """Load the held-out test split."""
    resolved = settings or get_settings()
    path = resolved.data_dir / "test" / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"test split not found at {path}. Run: python scripts/prepare_dataset.py"
        )
    return load_records(path)


def load_hard_test_records(settings: Settings | None = None) -> list[DatasetRecord]:
    """Load the hand-authored hard test set.

    The template test set is generated from the same grammar as the training
    data, so a keyword system can saturate it. The hard set is written by hand
    and contains paraphrased attacks with no signature vocabulary, benign events
    that deliberately contain alarming keywords, and inputs with genuinely
    insufficient evidence. It is where semantic understanding shows up.
    """
    resolved = settings or get_settings()
    path = resolved.data_dir / "test" / "hard_test.jsonl"
    if not path.exists():
        logger.warning(f"hard test set not found at {path}")
        return []
    return load_records(path)
