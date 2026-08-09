"""Retrieval pipeline.

    query -> normalisation -> embedding -> vector search -> optional rerank
          -> context assembly (with citations)

Two behaviours matter for grounding:

* Every retrieved chunk keeps its source metadata, and the assembled context
  string carries the identifiers verbatim so downstream grounding checks can
  verify that a claimed technique really appeared in the context.
* An empty result is returned as an empty result. The retriever never invents a
  document to fill the gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cybersentinel.cybersecurity.mitre import (
    CATEGORY_CWE_HINTS,
    CATEGORY_TECHNIQUE_HINTS,
    get_technique,
    get_weakness,
)
from cybersentinel.cybersecurity.taxonomy import AttackType
from cybersentinel.rag.embeddings import EmbeddingBackend, get_embedding_backend, tokenize
from cybersentinel.rag.vectorstore import VectorStore, build_vector_store
from cybersentinel.schemas.analysis import RetrievedDocument
from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

_NOISE_PATTERN = re.compile(r"[^\w\s.:/@-]")


@dataclass
class RetrievalResult:
    """Outcome of one retrieval call, including its own diagnostics."""

    documents: list[RetrievedDocument] = field(default_factory=list)
    query: str = ""
    latency_seconds: float = 0.0
    store: str = "unknown"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def context_text(self, max_chars: int = 4000) -> str:
        """Assemble retrieved chunks into a citation-carrying context block."""
        blocks: list[str] = []
        used = 0
        for index, document in enumerate(self.documents, start=1):
            header = f"[{index}] {document.source}"
            if document.document_id:
                header += f" | {document.document_id}"
            if document.title:
                header += f" | {document.title}"
            block = f"{header}\n{document.content}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def references(self) -> list[dict[str, str | None]]:
        """Deduplicated citations for the report."""
        seen: set[tuple[str, str | None]] = set()
        references: list[dict[str, str | None]] = []
        for document in self.documents:
            key = (document.source, document.document_id)
            if key in seen:
                continue
            seen.add(key)
            references.append(
                {
                    "source": document.source,
                    "document_id": document.document_id,
                    "title": document.title,
                    "url": document.url,
                }
            )
        return references


def normalize_query(text: str) -> str:
    """Trim noise and cap length so the query embeds cleanly."""
    cleaned = _NOISE_PATTERN.sub(" ", text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:1000]


def build_query(
    attack_type: AttackType,
    evidence: list[str] | None = None,
    raw_text: str = "",
) -> str:
    """Compose a retrieval query from the detection result.

    Category hint identifiers are included so that the relevant ATT&CK and CWE
    chunks are reachable even when the raw text shares no vocabulary with them.
    Including an id in the *query* does not ground it - grounding requires the
    id to come back in a retrieved document.
    """
    parts: list[str] = []

    if attack_type not in (AttackType.UNKNOWN, AttackType.BENIGN):
        parts.append(attack_type.value)
        parts.extend(CATEGORY_TECHNIQUE_HINTS.get(attack_type, ()))
        parts.extend(CATEGORY_CWE_HINTS.get(attack_type, ()))

    if evidence:
        parts.extend(evidence[:4])

    if raw_text:
        parts.append(" ".join(raw_text.split()[:60]))

    return normalize_query(" ".join(parts))


def lexical_overlap(query: str, document: RetrievedDocument) -> float:
    """Jaccard-style token overlap, used by the lightweight reranker."""
    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(document.content))
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def rerank(
    query: str,
    documents: list[RetrievedDocument],
    vector_weight: float = 0.7,
) -> list[RetrievedDocument]:
    """Blend vector score with lexical overlap.

    A cheap, dependency-free reranker. It reliably promotes the chunk that
    literally contains a queried identifier, which matters for citation
    correctness.
    """
    scored = [
        (vector_weight * document.score + (1 - vector_weight) * lexical_overlap(query, document), document)
        for document in documents
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        document.model_copy(update={"score": round(score, 4)}) for score, document in scored
    ]


class Retriever:
    """Embedding + vector-store retrieval with graceful degradation."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: EmbeddingBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_backend()
        self.store = store or build_vector_store(self.settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        use_rerank: bool = True,
    ) -> RetrievalResult:
        """Run one retrieval. Failures are captured, never raised at the caller."""
        import time

        normalised = normalize_query(query)
        if not normalised:
            return RetrievalResult(query=query, store=self.store.name, error="empty query")

        limit = top_k if top_k is not None else self.settings.top_k
        threshold = (
            score_threshold if score_threshold is not None else self.settings.rag_score_threshold
        )
        started = time.perf_counter()

        try:
            vector = self.embedder.embed_query(normalised)
            # Over-fetch so the reranker has candidates to reorder.
            fetch = limit * 3 if use_rerank else limit
            documents = self.store.search(vector, top_k=fetch, score_threshold=threshold)
            if use_rerank and documents:
                documents = rerank(normalised, documents)
            documents = documents[:limit]
            return RetrievalResult(
                documents=documents,
                query=normalised,
                latency_seconds=round(time.perf_counter() - started, 3),
                store=self.store.name,
            )
        except Exception as exc:
            logger.warning(f"retrieval failed: {type(exc).__name__}: {exc}")
            return RetrievalResult(
                query=normalised,
                latency_seconds=round(time.perf_counter() - started, 3),
                store=self.store.name,
                error=f"{type(exc).__name__}: {exc}",
            )

    def retrieve_for_detection(
        self,
        attack_type: AttackType,
        evidence: list[str] | None = None,
        raw_text: str = "",
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve threat intelligence for a detection result."""
        return self.retrieve(build_query(attack_type, evidence, raw_text), top_k=top_k)

    def catalogue_context(self, attack_type: AttackType) -> str:
        """Verified catalogue entries for a category, as a last-resort context.

        Used only when the vector store returns nothing. These entries come from
        the hand-verified catalogue in `cybersecurity.mitre`, so they are still
        real identifiers with real URLs - not model output.
        """
        lines: list[str] = []
        for technique_id in CATEGORY_TECHNIQUE_HINTS.get(attack_type, ()):
            technique = get_technique(technique_id)
            if technique:
                lines.append(
                    f"[catalogue] MITRE ATT&CK | {technique.technique_id} | {technique.name} "
                    f"(tactic: {technique.tactic}) {technique.url}"
                )
        for cwe_id in CATEGORY_CWE_HINTS.get(attack_type, ()):
            weakness = get_weakness(cwe_id)
            if weakness:
                lines.append(f"[catalogue] CWE | {weakness.cwe_id} | {weakness.name} {weakness.url}")
        return "\n".join(lines)

    def health(self) -> dict[str, object]:
        return {**self.store.health(), **self.embedder.info}
