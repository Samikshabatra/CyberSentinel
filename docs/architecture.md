# Architecture

## 1. Overview

CyberSentinel analyses a security event and produces a structured, explainable
incident report. Four capabilities are combined, each doing the job it is
actually good at:

| Capability | Role | Where |
|---|---|---|
| Fine-tuned LLM | Cybersecurity classification and evidence extraction | `llm/`, `agents/threat_detector.py` |
| RAG | Current, citable threat intelligence | `rag/`, `agents/threat_intelligence.py` |
| LangGraph | Stateful orchestration, routing, human-in-the-loop | `graph/` |
| Deterministic code | Risk scoring, validation, routing, persistence | `cybersecurity/`, `database/` |

## 2. Request flow

```
Analyst / SOC tool
        │
        ▼
FastAPI  (api/)  ── validation, error handling, OpenAPI
        │
        ▼
AnalysisService (service.py) ── incident memory, persistence
        │
        ▼
LangGraph workflow (graph/workflow.py)
        │
   ┌────┴─────────────────────────────────────────────┐
   │ input_classifier      deterministic format detection
   │        │  conditional: alert / email / url / log /
   │        │               vulnerability / multi_event
   │        ▼
   │ threat_detector       fine-tuned cybersecurity LLM
   │        │  conditional: skip retrieval for benign/unknown
   │        ▼
   │ threat_intelligence   Qdrant retrieval + grounding filter
   │        ▼
   │ correlation           shared indicators, kill-chain ordering
   │        ▼
   │ risk_assessment       likelihood × impact (pure Python)
   │        ▼
   │ response_recommendation  playbook or LLM, safety-filtered
   │        ▼
   │ approval_gate         is analyst sign-off required?
   │        │  conditional
   │        ├── not required ────────────────┐
   │        ▼                                │
   │ [INTERRUPT] human_approval              │
   │        │  APPROVE ──────────────────────┤
   │        │  REJECT  → escalation → response_recommendation (once)
   │        │  ESCALATE → escalation ────────┤
   │        ▼                                ▼
   │ incident_report ◄───────────────────────┘
   └──────────────────────────────────────────────────┘
        │
        ▼
PostgreSQL (incident history)   Streamlit SOC dashboard
```

## 3. Component responsibilities

### `utils/`
Configuration (`config.py`, environment-driven, no hardcoded paths), structured
logging with per-node run/incident/latency records and secret redaction
(`logging.py`), and input sanitisation plus indicator extraction
(`validation.py`).

### `cybersecurity/`
The domain layer, and deliberately free of LLM calls:

- `taxonomy.py` - the closed label set, with `Unknown` as a first-class outcome.
- `risk.py` - the likelihood × impact matrix and the approval policy.
- `mitre.py` - a hand-verified catalogue of ATT&CK techniques and CWEs, used as
  an allowlist so the model cannot introduce an identifier that does not exist.
- `correlation.py` - shared-indicator detection and kill-chain ordering.

### `llm/`
`prompts.py` centralises every prompt. `structured_output.py` implements the
JSON recovery ladder (direct → extracted → repaired → typed failure).
`model.py` provides two interchangeable backends:

- `mock` - a deterministic keyword-scoring analyst. It requires no GPU, makes
  the whole system runnable and testable anywhere, and serves as the `rules`
  baseline in the ablation study.
- `hf` - Transformers with optional 4-bit quantisation and a LoRA adapter.

### `rag/`
Loading (`loaders.py`), cleaning and chunking (`chunking.py`), embeddings
(`embeddings.py`), the vector store (`vectorstore.py`) and the retrieval
pipeline with reranking (`retriever.py`).

### `graph/`
`state.py` defines the typed `CyberState`; `nodes.py` adapts agents into nodes
and guarantees no node raises; `edges.py` holds the routing functions as pure
functions of state; `workflow.py` compiles the graph with a checkpointer and the
approval interrupt.

### `database/`
SQLAlchemy models, engine/session management and a repository that is the only
place SQL is written.

## 4. Design decisions

**Not everything is an LLM.** Routing, risk scoring, validation and correlation
are deterministic. This makes the outputs reproducible and directly defensible:
a risk score can be recomputed by hand from the rationale printed in the report.

**Retrieval decides threat-intelligence facts, the model does not.** A technique
or CVE is reported only when it appears in the retrieved context. Rejected
claims are recorded in `MitreMapping.rejected_claims`, which is what makes the
hallucination metric measurable rather than assumed.

**Every dependency has a fallback.** Qdrant unavailable → local vector store.
ML stack absent → mock backend. PostgreSQL absent → SQLite. The health endpoint
reports which path is live, so degradation is visible rather than silent.

**Nodes never raise.** `node_guard` converts any unhandled exception into an
entry in `errors` plus a trace record. A retrieval outage degrades the report;
it does not lose the analysis.

**The graph pauses, it does not proceed.** Approval is a real LangGraph
interrupt backed by a checkpointer. A paused run is resumable by a later HTTP
request, and the system never acts on its own.

## 5. Data flow for grounding

```
detection result ──► build_query() ──► embedding ──► vector search
                                                        │
                                          top-k + lexical rerank
                                                        │
                              context text (identifiers preserved verbatim)
                                                        │
        model proposes identifiers ─────────────────────┤
                                                        ▼
                          filter_grounded_techniques / filter_grounded_cves
                                          │                        │
                                     grounded                  rejected
                                          ▼                        ▼
                                    incident report        rejected_claims
```

## 6. Deployment

- **Local, zero services**: mock backend, hash embeddings, local vector store,
  SQLite. One command runs the API or the UI.
- **Local with services**: `docker compose up -d qdrant postgres`.
- **Full container stack**: `docker compose --profile full up -d`.

Training runs on the host, not in a container, and produces a LoRA adapter that
is mounted read-only into the API container.
