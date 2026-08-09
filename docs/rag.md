# Retrieval-augmented generation

## 1. Purpose

RAG supplies the threat-intelligence facts the model is explicitly not trusted
to know: what a technique is, which weakness a payload maps to, and what
authoritative guidance says about response. Every fact carries a citation.

The grounding rule enforced in code:

> A MITRE technique, CWE or CVE identifier is reported **only** if it appears in
> the retrieved context. Anything else the model proposes is recorded as a
> rejected claim and shown in the report.

## 2. Knowledge base

`data/knowledge_base/` holds curated JSONL, one document per line, each carrying
its own source, identifier and official URL - so citations are copied from the
corpus rather than generated.

| File | Documents | Source | Licence / terms |
|---|---|---|---|
| `mitre_attack.jsonl` | 40 techniques | MITRE ATT&CK Enterprise | ATT&CK Terms of Use, attribution to MITRE |
| `cwe.jsonl` | 16 weaknesses | MITRE CWE | Free to use with attribution to MITRE |
| `guidance.jsonl` | 10 documents | NIST SP 800-61 / 800-63B / 800-83, OWASP cheat sheets and Top 10, CISA advisories | NIST publications are public domain; OWASP content is CC BY-SA; CISA content is public domain |

Entries are summaries written for this project from the public documentation,
with the identifier and canonical URL preserved so a reader can verify any claim
at the source. They are not verbatim copies.

Optional live ingestion of the official ATT&CK STIX bundle is supported:

```bash
python scripts/ingest_knowledge_base.py --fetch-attack
```

This downloads `enterprise-attack.json` from MITRE's own `mitre/cti` repository
and parses every non-revoked, non-deprecated technique. It is optional so the
project stays reproducible offline; failure is logged and the offline corpus is
used.

## 3. Ingestion pipeline

```
load JSONL ──► clean (strip HTML, citation markers, whitespace)
           ──► chunk (paragraph-aware, 900 chars, 120 overlap)
           ──► prepend provenance header
           ──► deduplicate on normalised text
           ──► embed
           ──► upsert into the vector store
```

```bash
python scripts/ingest_knowledge_base.py            # Qdrant, falls back to local
python scripts/ingest_knowledge_base.py --local    # local store only
python scripts/ingest_knowledge_base.py --reset    # rebuild from scratch
```

### The provenance header matters

Every chunk is prefixed with `source | document_id | title | tactic`. This is not
cosmetic:

1. It puts the identifier (`T1110`, `CWE-89`) into the **embedded text**, so a
   query mentioning an identifier can actually match the chunk describing it.
2. It makes a retrieved fragment self-describing in the UI.
3. It is what the grounding filter checks against - the identifier must come
   back from retrieval, not from the model.

During development, retrieval returned nothing useful until this was added: the
identifiers lived only in metadata, so an identifier query had nothing to match.

## 4. Embeddings

Two interchangeable backends:

| Backend | When | Trade-off |
|---|---|---|
| `hash` (default) | Anywhere. No model download, no GPU, deterministic. | Weaker semantics; lower absolute similarity scores. |
| `sentence-transformers` | With the `ml` extra installed. | Better recall on paraphrases; needs a model download. |

The hash backend projects hashed token and bigram features into a fixed vector
with sublinear term weighting and L2 normalisation. Threat-intelligence
identifiers are weighted three times higher than ordinary tokens, because in
this domain an identifier is the highest-signal token in the text.

Because hash scores are lower in absolute terms, `RAG_SCORE_THRESHOLD` defaults
to `0.05`. Raise it to about `0.25` when switching to sentence-transformers.

## 5. Vector store

Qdrant is the primary store (cosine distance, deterministic UUID point ids so
re-ingestion updates in place rather than duplicating).

If Qdrant is unreachable, the retriever falls back to `LocalVectorStore`, a
brute-force cosine search over a JSON file. For a knowledge base of this size
this is entirely adequate, and it means retrieval degrades rather than fails.
Which store served a query is reported in `RetrievalResult.store` and in
`GET /health`.

## 6. Retrieval pipeline

```
query ──► normalise ──► embed ──► vector search (top_k × 3)
      ──► rerank (0.7 × vector score + 0.3 × lexical overlap)
      ──► top_k ──► context assembly with citations
```

The query is composed from the detection result: the category name, the
category's candidate technique and CWE identifiers, the extracted evidence and a
prefix of the raw text. Including an identifier in the *query* does not ground
it - grounding requires the identifier to come back in a retrieved document.

The reranker is cheap and dependency-free, and it reliably promotes the chunk
that literally contains a queried identifier, which is what citation correctness
depends on.

## 7. Grounding and rejection

```python
grounded, rejected = filter_grounded_techniques(candidates, context_text)
accepted, rejected_cves = filter_grounded_cves(candidates, context_text)
```

A candidate technique is accepted only if it is **both** present in the
hand-verified catalogue (`cybersecurity/mitre.py`) **and** present in the
retrieved context. A CVE is accepted only if it is syntactically valid and
appears in the retrieved context - the model may never introduce a CVE on its
own, because there is no catalogue of all valid CVEs to check against.

Rejected identifiers are surfaced in the report rather than silently dropped.
That is what makes the hallucination rate measurable.

If retrieval returns nothing, the system falls back to the verified catalogue
for the detected category. Those identifiers are still real - they come from the
catalogue, not the model - but `grounded` stays `false` so the report and the
evaluation can tell the difference.

## 8. Configuration

`configs/rag.yaml` documents the pipeline; environment variables override it:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cybersentinel_kb
EMBEDDING_BACKEND=hash
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
TOP_K=5
RAG_SCORE_THRESHOLD=0.05
```

## 9. Measured performance

From `scripts/evaluate.py` on the current build (hash embeddings, local store,
66 chunks):

| Metric | Value |
|---|---|
| Mean precision@5 | 0.77 |
| Mean recall@5 | 0.96 |
| Mean reciprocal rank | 1.00 |
| Grounded rate with RAG | 1.00 |
| Grounded rate without RAG | 0.00 |
| MITRE mapping recall with RAG | 1.00 |

Mean reciprocal rank of 1.00 means the correct document was ranked first for
every category. Precision is below 1.0 because a top-5 query returns related
documents (a brute-force query legitimately also returns CWE-307 and password
guidance) that are not in the narrow ground-truth set for that category.

## 10. Limitations

- The knowledge base is a curated subset, not a complete threat-intelligence
  feed. There is no live CVE ingestion by default.
- The default hash embedding is weaker than a trained encoder on paraphrases.
- Relevance ground truth for the retrieval study is derived from the taxonomy's
  category-to-technique mapping, not from human annotation.
- Nothing in the pipeline fetches URLs found in analyst input. Only MITRE's own
  distribution repository is ever contacted, and only on explicit request.
