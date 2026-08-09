"""RAG tests: chunking, embeddings, retrieval, grounding filters, degradation."""

from __future__ import annotations

import pytest

from cybersentinel.cybersecurity import mitre
from cybersentinel.cybersecurity.taxonomy import AttackType
from cybersentinel.rag.chunking import (
    Document,
    build_header,
    chunk_document,
    clean_text,
    deduplicate_chunks,
)
from cybersentinel.rag.embeddings import HashEmbedding, cosine_similarity, tokenize
from cybersentinel.rag.loaders import parse_attack_stix
from cybersentinel.rag.retriever import build_query, normalize_query, rerank
from cybersentinel.rag.vectorstore import LocalVectorStore


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_clean_text_strips_markup_and_citations():
    dirty = "<p>Adversaries do things.</p> (Citation: Some Source 2020)\n\n\n\nNext paragraph."
    cleaned = clean_text(dirty)
    assert "<p>" not in cleaned
    assert "Citation" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_carries_identifier_into_the_text():
    """Grounding depends on the identifier being present in the retrieved text."""
    document = Document(
        content="Adversaries guess passwords repeatedly.",
        metadata={"source": "MITRE ATT&CK", "document_id": "T1110", "title": "Brute Force"},
    )
    chunks = chunk_document(document)

    assert chunks
    assert "T1110" in chunks[0].content
    assert chunks[0].metadata["document_id"] == "T1110"


def test_long_document_splits_into_multiple_chunks():
    document = Document(
        content="\n\n".join(f"Paragraph {index}. " + "word " * 120 for index in range(6)),
        metadata={"source": "test", "document_id": "LONG-1", "title": "Long"},
    )
    chunks = chunk_document(document, chunk_size=400, overlap=50)

    assert len(chunks) > 1
    assert all(chunk.metadata["document_id"] == "LONG-1" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_point_ids_are_unique():
    document = Document(
        content="\n\n".join("x " * 200 for _ in range(4)),
        metadata={"source": "test", "document_id": "P-1"},
    )
    chunks = chunk_document(document, chunk_size=300, overlap=20)
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)


def test_duplicate_chunks_are_removed():
    chunks = chunk_document(
        Document(content="Same text.", metadata={"source": "s", "document_id": "A"})
    ) + chunk_document(Document(content="Same text.", metadata={"source": "s", "document_id": "A"}))
    assert len(deduplicate_chunks(chunks)) == 1


def test_header_includes_provenance():
    header = build_header({"source": "CWE", "document_id": "CWE-89", "title": "SQL Injection"})
    assert "CWE-89" in header
    assert "SQL Injection" in header


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_embeddings_are_deterministic():
    embedder = HashEmbedding(128)
    assert embedder.embed_query("brute force") == embedder.embed_query("brute force")


def test_embedding_dimension_is_respected():
    assert len(HashEmbedding(64).embed_query("test")) == 64


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = HashEmbedding(384)
    query = embedder.embed_query("failed SSH login attempts brute force")
    related = embedder.embed_query("repeated failed login attempts indicate brute force")
    unrelated = embedder.embed_query("scheduled backup completed successfully overnight")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_identifiers_survive_tokenisation():
    tokens = tokenize("Technique T1110 relates to CWE-307 and CVE-2024-12345")
    assert "T1110" in tokens
    assert "CWE-307" in tokens
    assert "CVE-2024-12345" in tokens


def test_empty_text_embeds_to_zero_vector():
    assert set(HashEmbedding(32).embed_query("")) == {0.0}


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def test_local_store_round_trip(tmp_path):
    store = LocalVectorStore(tmp_path / "store.json")
    embedder = HashEmbedding(128)

    document = Document(
        content="Adversaries guess passwords repeatedly against one account.",
        metadata={"source": "MITRE ATT&CK", "document_id": "T1110", "title": "Brute Force"},
    )
    chunks = chunk_document(document)
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    store.ensure_collection(128)
    store.upsert(chunks, vectors)

    results = store.search(embedder.embed_query("password guessing T1110"), top_k=1)

    assert store.count() == len(chunks)
    assert results
    assert results[0].document_id == "T1110"
    assert results[0].url is None or isinstance(results[0].url, str)


def test_reingestion_updates_in_place(tmp_path):
    store = LocalVectorStore(tmp_path / "store.json")
    embedder = HashEmbedding(64)
    chunks = chunk_document(
        Document(content="text", metadata={"source": "s", "document_id": "D1"})
    )
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])

    store.ensure_collection(64)
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)

    assert store.count() == len(chunks)


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "store.json"
    embedder = HashEmbedding(64)
    chunks = chunk_document(Document(content="text", metadata={"source": "s", "document_id": "D1"}))

    first = LocalVectorStore(path)
    first.ensure_collection(64)
    first.upsert(chunks, embedder.embed_documents([chunk.content for chunk in chunks]))

    assert LocalVectorStore(path).count() == first.count()


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_query_normalisation_trims_noise():
    assert normalize_query("  brute   force!!!  \n attack  ") == "brute force attack"


def test_query_includes_category_identifiers():
    query = build_query(AttackType.BRUTE_FORCE, ["failed logins"], "47 failures")
    assert "T1110" in query
    assert "Brute Force" in query


def test_query_for_unknown_omits_identifiers():
    assert build_query(AttackType.UNKNOWN, [], "vague text").find("T1") == -1


@pytest.mark.parametrize(
    ("category", "expected_id"),
    [
        (AttackType.BRUTE_FORCE, "T1110"),
        (AttackType.SQL_INJECTION, "CWE-89"),
        (AttackType.PHISHING, "T1566"),
        (AttackType.DATA_EXFILTRATION, "T1041"),
    ],
)
def test_retrieval_finds_the_expected_document(retriever, category, expected_id):
    result = retriever.retrieve_for_detection(category, [], "", top_k=8)
    retrieved = {document.document_id for document in result.documents}

    assert result.ok
    assert any(str(document_id).startswith(expected_id) for document_id in retrieved), retrieved


def test_retrieved_documents_carry_citations(retriever):
    result = retriever.retrieve_for_detection(AttackType.BRUTE_FORCE, [], "", top_k=3)
    for document in result.documents:
        assert document.source
        assert document.url


def test_context_text_contains_identifiers(retriever):
    result = retriever.retrieve_for_detection(AttackType.BRUTE_FORCE, [], "", top_k=5)
    assert "T1110" in result.context_text()


def test_references_are_deduplicated(retriever):
    result = retriever.retrieve_for_detection(AttackType.BRUTE_FORCE, [], "", top_k=8)
    references = result.references()
    keys = [(reference["source"], reference["document_id"]) for reference in references]
    assert len(keys) == len(set(keys))


def test_empty_query_is_reported_not_raised(retriever):
    result = retriever.retrieve("   ")
    assert not result.ok
    assert result.is_empty


def test_empty_store_returns_no_documents(empty_retriever):
    result = empty_retriever.retrieve("brute force")
    assert result.is_empty
    assert result.ok


def test_rerank_promotes_the_literal_identifier_match():
    from cybersentinel.schemas.analysis import RetrievedDocument

    documents = [
        RetrievedDocument(content="general authentication guidance", score=0.5, source="s"),
        RetrievedDocument(content="MITRE ATT&CK T1110 Brute Force", score=0.45, source="s"),
    ]
    reranked = rerank("T1110 brute force", documents)
    assert "T1110" in reranked[0].content


# --------------------------------------------------------------------------- #
# Grounding filters
# --------------------------------------------------------------------------- #
def test_only_catalogued_techniques_are_accepted():
    grounded, rejected = mitre.filter_grounded_techniques(["T1110", "T9999"], "context has T1110")
    assert [technique.technique_id for technique in grounded] == ["T1110"]
    assert rejected == ["T9999"]


def test_technique_absent_from_context_is_rejected():
    grounded, rejected = mitre.filter_grounded_techniques(["T1110"], "context mentions T1566 only")
    assert grounded == []
    assert rejected == ["T1110"]


def test_cve_must_appear_in_context():
    accepted, rejected = mitre.filter_grounded_cves(
        ["CVE-2024-12345", "CVE-2099-99999"], "advisory covers CVE-2024-12345"
    )
    assert accepted == ["CVE-2024-12345"]
    assert rejected == ["CVE-2099-99999"]


@pytest.mark.parametrize("identifier", ["CVE-2024-1234", "CVE-2024-1234567"])
def test_wellformed_cve_ids(identifier):
    assert mitre.is_wellformed_cve_id(identifier)


@pytest.mark.parametrize("identifier", ["CVE-24-1234", "CVE-2024-123", "T1110", "nonsense"])
def test_malformed_cve_ids(identifier):
    assert not mitre.is_wellformed_cve_id(identifier)


def test_extract_identifiers_from_free_text():
    found = mitre.extract_identifiers("Maps to T1110.003 and CWE-307, referencing CVE-2024-12345.")
    assert found["techniques"] == ["T1110.003"]
    assert found["cwes"] == ["CWE-307"]
    assert found["cves"] == ["CVE-2024-12345"]


def test_kill_chain_ordering_is_narrative():
    techniques = [mitre.TECHNIQUES["T1068"], mitre.TECHNIQUES["T1595"], mitre.TECHNIQUES["T1110"]]
    ordered = [technique.technique_id for technique in mitre.order_by_kill_chain(techniques)]
    assert ordered == ["T1595", "T1110", "T1068"]


def test_catalogue_entries_have_valid_urls():
    for technique in mitre.TECHNIQUES.values():
        assert technique.url.startswith("https://attack.mitre.org/techniques/")
    for weakness in mitre.WEAKNESSES.values():
        assert weakness.url.startswith("https://cwe.mitre.org/data/definitions/")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def test_stix_parser_extracts_techniques():
    bundle = {
        "objects": [
            {"type": "x-mitre-tactic", "x_mitre_shortname": "credential-access",
             "name": "Credential Access"},
            {
                "type": "attack-pattern",
                "name": "Brute Force",
                "description": "Adversaries guess passwords.",
                "kill_chain_phases": [
                    {"kill_chain_name": "mitre-attack", "phase_name": "credential-access"}
                ],
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1110",
                     "url": "https://attack.mitre.org/techniques/T1110/"}
                ],
            },
            {
                "type": "attack-pattern",
                "name": "Revoked",
                "revoked": True,
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T0000", "url": "x"}
                ],
            },
        ]
    }
    documents = parse_attack_stix(bundle)

    assert len(documents) == 1
    assert documents[0].metadata["document_id"] == "T1110"
    assert "Credential Access" in documents[0].content
