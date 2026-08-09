"""Knowledge-base loaders.

Two ingestion paths:

* **Offline (default)** - curated JSONL files under ``data/knowledge_base/``.
  Each record carries its own source, identifier and official URL, so citations
  are copied from the corpus rather than generated.
* **Online (optional)** - the official MITRE ATT&CK STIX bundle from the
  ``mitre/cti`` repository. Used when the machine has network access; the
  offline corpus keeps the project reproducible without it.

No loader fetches arbitrary URLs found inside analyst input.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cybersentinel.rag.chunking import Document
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

#: Official ATT&CK Enterprise STIX bundle (MITRE's own distribution repository).
MITRE_ENTERPRISE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)

REQUIRED_FIELDS = ("content", "source")


class KnowledgeBaseError(RuntimeError):
    """Raised when a knowledge-base file cannot be loaded."""


def load_jsonl(path: Path) -> list[Document]:
    """Load a JSONL knowledge-base file into documents."""
    if not path.exists():
        raise KnowledgeBaseError(f"knowledge-base file not found: {path}")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(f"{path.name}:{line_number} skipped - invalid JSON ({exc.msg})")
                continue

            missing = [field for field in REQUIRED_FIELDS if not record.get(field)]
            if missing:
                logger.warning(f"{path.name}:{line_number} skipped - missing {missing}")
                continue

            content = record.pop("content")
            documents.append(Document(content=content, metadata=record))

    logger.info(f"loaded {len(documents)} documents from {path.name}")
    return documents


def load_directory(directory: Path, pattern: str = "*.jsonl") -> list[Document]:
    """Load every matching knowledge-base file in a directory."""
    if not directory.exists():
        raise KnowledgeBaseError(f"knowledge-base directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.glob(pattern)):
        documents.extend(load_jsonl(path))

    if not documents:
        raise KnowledgeBaseError(f"no documents found in {directory} matching {pattern}")
    return documents


def _stix_external_reference(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (attack id, url) from a STIX object's external references."""
    for reference in obj.get("external_references", []):
        if reference.get("source_name") == "mitre-attack":
            return reference.get("external_id"), reference.get("url")
    return None, None


def parse_attack_stix(bundle: dict[str, Any]) -> list[Document]:
    """Convert an ATT&CK STIX bundle into knowledge-base documents.

    Only non-revoked, non-deprecated ``attack-pattern`` objects are kept.
    """
    tactic_names = {
        obj.get("x_mitre_shortname"): obj.get("name")
        for obj in bundle.get("objects", [])
        if obj.get("type") == "x-mitre-tactic"
    }

    documents: list[Document] = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        attack_id, url = _stix_external_reference(obj)
        if not attack_id:
            continue

        tactics = [
            tactic_names.get(phase.get("phase_name"), phase.get("phase_name"))
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]

        detection = obj.get("x_mitre_detection", "")
        platforms = ", ".join(obj.get("x_mitre_platforms", []))
        body = obj.get("description", "")
        if tactics:
            body += f"\n\nTactics: {', '.join(tactics)}."
        if platforms:
            body += f"\nPlatforms: {platforms}."
        if detection:
            body += f"\n\nDetection: {detection}"

        documents.append(
            Document(
                content=body,
                metadata={
                    "source": "MITRE ATT&CK",
                    "document_id": attack_id,
                    "title": obj.get("name"),
                    "url": url,
                    "category": "attack-pattern",
                    "tactics": tactics,
                },
            )
        )

    logger.info(f"parsed {len(documents)} ATT&CK techniques from STIX bundle")
    return documents


def fetch_attack_stix(url: str = MITRE_ENTERPRISE_STIX_URL, timeout: int = 120) -> list[Document]:
    """Download and parse the official ATT&CK Enterprise STIX bundle.

    Network access is required. Callers must treat failure as non-fatal and fall
    back to the offline corpus.
    """
    import httpx

    logger.info(f"downloading ATT&CK STIX bundle from {url}")
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return parse_attack_stix(response.json())


def load_cve_records(path: Path) -> list[Document]:
    """Load CVE records exported from NVD into documents.

    Expected shape per line: ``{"cve_id", "description", "cvss_score",
    "published", "references"}``. Nothing is invented - if a field is absent it
    is simply omitted from the text.
    """
    if not path.exists():
        return []

    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            cve_id = record.get("cve_id")
            description = record.get("description")
            if not cve_id or not description:
                continue

            body = description
            if record.get("cvss_score") is not None:
                body += f"\n\nCVSS base score: {record['cvss_score']}."
            if record.get("published"):
                body += f"\nPublished: {record['published']}."
            if record.get("cwe"):
                body += f"\nAssociated weakness: {record['cwe']}."

            documents.append(
                Document(
                    content=body,
                    metadata={
                        "source": "NVD",
                        "document_id": cve_id,
                        "title": cve_id,
                        "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        "category": "vulnerability",
                    },
                )
            )
    return documents


def summarise_corpus(documents: Iterable[Document]) -> dict[str, int]:
    """Count documents per source - printed by the ingestion script."""
    counts: dict[str, int] = {}
    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts
