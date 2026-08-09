"""Document cleaning and chunking.

Knowledge-base entries are short and structured, so chunking is paragraph-aware
with a character budget and overlap. Metadata is copied onto every chunk so a
retrieved fragment always carries its provenance - citations are never
reconstructed later from memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HTML_TAG = re.compile(r"<[^>]+>")
_CITATION_MARKER = re.compile(r"\(Citation:[^)]*\)")


@dataclass
class Document:
    """A knowledge-base document before chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str | None:
        value = self.metadata.get("document_id")
        return str(value) if value is not None else None


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0

    @property
    def point_id(self) -> str:
        """Stable identifier: document id plus chunk index."""
        base = self.metadata.get("document_id") or self.metadata.get("title") or "doc"
        return f"{base}::{self.chunk_index}"


def clean_text(text: str) -> str:
    """Normalise whitespace, strip HTML tags and ATT&CK citation markers."""
    cleaned = _HTML_TAG.sub(" ", text)
    cleaned = _CITATION_MARKER.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _WHITESPACE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return "\n".join(line.strip() for line in cleaned.split("\n")).strip()


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            # A single sentence longer than the budget is hard-split.
            while len(sentence) > chunk_size:
                pieces.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    document: Document,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Split one document into overlapping, metadata-carrying chunks."""
    cleaned = clean_text(document.content)
    if not cleaned:
        return []

    paragraphs = _split_paragraphs(cleaned)

    blocks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            blocks.append(paragraph)
        else:
            blocks.extend(_split_long_paragraph(paragraph, chunk_size))

    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{block}".strip() if tail else block
            else:
                current = block

    if current:
        chunks.append(current)

    # Every chunk carries a provenance header. This makes a retrieved fragment
    # self-describing, and - importantly - puts the identifier (T1110, CWE-89)
    # into the embedded text, so an identifier query can actually match it and
    # so grounding checks can verify the identifier came back from retrieval.
    header = build_header(document.metadata)
    return [
        Chunk(
            content=f"{header}\n{content}" if header else content,
            metadata=dict(document.metadata),
            chunk_index=index,
        )
        for index, content in enumerate(chunks)
    ]


def build_header(metadata: dict[str, Any]) -> str:
    """Build the provenance header prefixed to every chunk."""
    parts = [
        str(metadata[key])
        for key in ("source", "document_id", "title")
        if metadata.get(key)
    ]
    tactic = metadata.get("tactic")
    if tactic:
        parts.append(f"tactic: {tactic}")
    return " | ".join(parts)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk a corpus."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    return chunks


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks with identical normalised content."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = " ".join(chunk.content.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
