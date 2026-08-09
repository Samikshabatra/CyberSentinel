"""Threat-intelligence agent (LangGraph node 3).

Retrieval decides what is true here, not the model. The agent:

1. retrieves from the knowledge base using the detection result,
2. asks the model which identifiers the retrieved context supports,
3. discards every identifier that is not both in the verified catalogue and
   present in the retrieved text.

Discarded identifiers are recorded as ``rejected_claims`` so the hallucination
metrics can count them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cybersentinel.cybersecurity import mitre
from cybersentinel.cybersecurity.taxonomy import AttackType
from cybersentinel.llm.inference import generate
from cybersentinel.llm.model import LLMBackend
from cybersentinel.llm.prompts import build_intel_messages
from cybersentinel.llm.structured_output import parse_json_object
from cybersentinel.rag.retriever import RetrievalResult, Retriever
from cybersentinel.schemas.analysis import MitreMapping, SourceReference, ThreatAnalysis
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntelOutcome:
    """Grounded mapping plus the retrieval that produced it."""

    mapping: MitreMapping
    retrieval: RetrievalResult
    sources: list[SourceReference] = field(default_factory=list)
    context_text: str = ""
    latency_seconds: float = 0.0
    used_catalogue_fallback: bool = False
    error: str | None = None


def _catalogue_only_mapping(attack_type: AttackType, reason: str) -> MitreMapping:
    """Mapping built purely from the verified catalogue, clearly not RAG-grounded.

    Used when retrieval is unavailable. The identifiers are still real (they come
    from the hand-verified catalogue, not the model), but ``grounded`` stays
    False so the report and the evaluation can tell the difference.
    """
    techniques = [
        technique.to_dict()
        for technique_id in mitre.CATEGORY_TECHNIQUE_HINTS.get(attack_type, ())
        if (technique := mitre.get_technique(technique_id)) is not None
    ][:3]
    weaknesses = [
        weakness.to_dict()
        for cwe_id in mitre.CATEGORY_CWE_HINTS.get(attack_type, ())
        if (weakness := mitre.get_weakness(cwe_id)) is not None
    ][:2]
    return MitreMapping(
        techniques=techniques,
        tactics=sorted({technique["tactic"] for technique in techniques}),
        cwe=weaknesses,
        cve=[],
        rejected_claims=[],
        grounded=False,
    )


def gather_intelligence(
    analysis: ThreatAnalysis,
    raw_text: str,
    retriever: Retriever,
    backend: LLMBackend | None = None,
    use_llm: bool = True,
    top_k: int | None = None,
) -> IntelOutcome:
    """Retrieve threat intelligence and produce a grounded mapping."""
    import time

    started = time.perf_counter()

    if analysis.attack_type in (AttackType.BENIGN, AttackType.UNKNOWN):
        # Nothing to map. Retrieval is skipped rather than run on a non-finding.
        return IntelOutcome(
            mapping=MitreMapping(grounded=False),
            retrieval=RetrievalResult(query="", store=retriever.store.name),
            context_text="",
            latency_seconds=round(time.perf_counter() - started, 3),
        )

    retrieval = retriever.retrieve_for_detection(
        analysis.attack_type, analysis.evidence, raw_text, top_k=top_k
    )
    context_text = retrieval.context_text()
    used_fallback = False

    if retrieval.is_empty:
        # Vector store empty or unavailable: fall back to the verified catalogue
        # so the analyst still gets real references, marked as ungrounded.
        logger.warning("retrieval returned no documents; using catalogue fallback")
        mapping = _catalogue_only_mapping(analysis.attack_type, retrieval.error or "no results")
        return IntelOutcome(
            mapping=mapping,
            retrieval=retrieval,
            sources=[
                SourceReference(
                    source=technique["source"],
                    document_id=technique["technique_id"],
                    title=technique["name"],
                    url=technique["url"],
                    category="attack-pattern",
                )
                for technique in mapping.techniques
            ],
            context_text=retriever.catalogue_context(analysis.attack_type),
            latency_seconds=round(time.perf_counter() - started, 3),
            used_catalogue_fallback=True,
            error=retrieval.error,
        )

    # Candidates: what the detector proposed, plus anything the context mentions.
    context_identifiers = mitre.extract_identifiers(context_text)
    candidate_techniques = list(analysis.candidate_techniques)
    candidate_cwes: list[str] = list(context_identifiers["cwes"])
    candidate_cves: list[str] = list(context_identifiers["cves"])

    if use_llm:
        messages = build_intel_messages(
            analysis.attack_type.value, analysis.severity.value, analysis.evidence, context_text
        )
        result = generate(messages, backend=backend)
        if result.ok:
            parsed = parse_json_object(result.text)
            if parsed.ok and parsed.data:
                candidate_techniques += [
                    str(item) for item in parsed.data.get("techniques", []) if item
                ]
                candidate_cwes += [str(item) for item in parsed.data.get("cwe", []) if item]
                candidate_cves += [str(item) for item in parsed.data.get("cve", []) if item]
        else:
            logger.warning(f"intel model call failed: {result.error}")

    # Anything the context itself asserts is grounded by definition.
    candidate_techniques += context_identifiers["techniques"]

    grounded_techniques, rejected = mitre.filter_grounded_techniques(
        candidate_techniques, context_text
    )
    grounded_cves, rejected_cves = mitre.filter_grounded_cves(candidate_cves, context_text)

    grounded_weaknesses = []
    for cwe_id in candidate_cwes:
        normalised = cwe_id.strip().upper()
        weakness = mitre.get_weakness(normalised)
        if weakness and normalised in context_text.upper():
            if weakness.to_dict() not in grounded_weaknesses:
                grounded_weaknesses.append(weakness.to_dict())
        elif normalised not in rejected:
            rejected.append(normalised)

    mapping = MitreMapping(
        techniques=[technique.to_dict() for technique in mitre.order_by_kill_chain(grounded_techniques)],
        tactics=sorted({technique.tactic for technique in grounded_techniques}),
        cwe=grounded_weaknesses,
        cve=[mitre.cve_reference(cve_id) for cve_id in grounded_cves],
        rejected_claims=sorted(set(rejected + rejected_cves)),
        grounded=bool(grounded_techniques or grounded_weaknesses or grounded_cves),
    )

    if mapping.rejected_claims:
        logger.info(
            f"rejected {len(mapping.rejected_claims)} ungrounded identifier(s): "
            f"{mapping.rejected_claims}"
        )

    return IntelOutcome(
        mapping=mapping,
        retrieval=retrieval,
        sources=[document.to_reference() for document in retrieval.documents],
        context_text=context_text,
        latency_seconds=round(time.perf_counter() - started, 3),
        used_catalogue_fallback=used_fallback,
        error=retrieval.error,
    )
