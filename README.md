# CyberSentinel

**An Agentic LLM Framework for Cybersecurity Threat Detection, Intelligence
Retrieval and Incident Response**

CyberSentinel takes a security event - an alert, a log excerpt, a suspicious
email, a URL, or several related events - and produces a structured, explainable
incident report: what was detected, why, which evidence supports it, what the
retrieved threat intelligence says, how risky it is, and what an analyst should
do next.

It is an **AI-assisted analysis system**. Every output is a recommendation for a
human. Nothing is executed automatically.

---

## Table of contents

1. [Problem](#1-problem)
2. [What it does](#2-what-it-does)
3. [Architecture](#3-architecture)
4. [Technology stack](#4-technology-stack)
5. [Quick start](#5-quick-start)
6. [Running the system](#6-running-the-system)
7. [Dataset](#7-dataset)
8. [Fine-tuning](#8-fine-tuning-qlora)
9. [RAG pipeline](#9-rag-pipeline)
10. [LangGraph workflow](#10-langgraph-workflow)
11. [Agents](#11-agents)
12. [API](#12-api)
13. [Evaluation](#13-evaluation)
14. [Demo scenarios](#14-demo-scenarios)
15. [Testing](#15-testing)
16. [Project structure](#16-project-structure)
17. [Security considerations](#17-security-considerations)
18. [Limitations](#18-limitations)
19. [Future work](#19-future-work)

---

## 1. Problem

Security analysts receive heterogeneous events - phishing emails, authentication
logs, network alerts, malware descriptions, injection payloads, vulnerability
reports. Rule-based systems detect known patterns but struggle with
natural-language interpretation, cross-source correlation, and explaining a
conclusion in terms an analyst can act on.

An LLM can interpret language, but on its own it will confidently invent a CVE
identifier that does not exist. That is worse than saying nothing.

CyberSentinel addresses both halves: a cybersecurity-specialised model does the
reasoning, retrieval supplies the citable facts, deterministic code does the
scoring and routing, and a human approves anything consequential.

## 2. What it does

1. Accepts a security event (or several, for correlation).
2. Classifies the input format and routes accordingly.
3. Detects the likely threat with a cybersecurity-specialised LLM.
4. Retrieves supporting threat intelligence from MITRE ATT&CK, CWE and public
   guidance.
5. Maps findings to identifiers **only when retrieval supports them**.
6. Correlates related events into a possible attack chain.
7. Scores risk with a transparent likelihood × impact matrix.
8. Recommends defensive actions, filtered for safety.
9. **Pauses for analyst approval** on high-risk or operationally impactful
   recommendations.
10. Produces a structured, explainable report and stores it as incident history.

## 3. Architecture

```
Analyst / SOC tool
        │
        ▼
   FastAPI  ──► AnalysisService ──► LangGraph workflow
                      │                     │
                      │        ┌────────────┴─────────────┐
                      │        │  input_classifier        │  deterministic
                      │        │  threat_detector         │  fine-tuned LLM
                      │        │  threat_intelligence     │  RAG + grounding
                      │        │  correlation             │  deterministic
                      │        │  risk_assessment         │  deterministic
                      │        │  response_recommendation │  LLM + safety filter
                      │        │  approval_gate           │  policy
                      │        │  [INTERRUPT] human_approval
                      │        │  incident_report         │
                      │        └──────────────────────────┘
                      ▼
              PostgreSQL / SQLite            Qdrant / local vector store
              (incident history)             (knowledge base)
                      │
                      ▼
              Streamlit SOC dashboard
```

Full detail: [`docs/architecture.md`](docs/architecture.md).

**Design principle:** not every component is an LLM. Routing, risk scoring,
validation, correlation and persistence are deterministic Python, which is why
their outputs are reproducible and defensible. The model is used where language
understanding is genuinely required.

## 4. Technology stack

| Layer | Choice |
|---|---|
| LLM | Qwen2.5-3B-Instruct (configurable), QLoRA fine-tuned |
| Fine-tuning | Transformers, PEFT, TRL, bitsandbytes, Accelerate |
| Orchestration | LangGraph (stateful, conditional, checkpointed) |
| RAG | Qdrant, sentence-transformers or a dependency-free hash embedder |
| Backend | FastAPI, Pydantic v2 |
| Database | PostgreSQL (SQLite by default) via SQLAlchemy 2.0 |
| UI | Streamlit |
| Runtime | Python 3.11+, Docker Compose |

## 5. Quick start

Runs with **no GPU, no Docker and no API keys**. Fallbacks are built in: a
deterministic rules backend, a local vector store and SQLite.

```bash
git clone <repository-url>
cd LLM_Project

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev,ui,eval]"
cp .env.example .env              # optional; defaults work as-is

python scripts/prepare_dataset.py
python scripts/ingest_knowledge_base.py --local

streamlit run app/streamlit_app.py    # http://localhost:8501
```

The dashboard ships with six demo scenarios; pick one and press **Analyse**.

## 6. Running the system

### API

```bash
uvicorn cybersentinel.api.main:app --reload
# http://localhost:8000/docs
```

### UI

```bash
streamlit run app/streamlit_app.py
```

The UI uses the API when it is reachable and falls back to running the pipeline
in-process when it is not, so either command works on its own.

### With Qdrant and PostgreSQL

```bash
docker compose up -d qdrant postgres
python scripts/ingest_knowledge_base.py
```

Then set in `.env`:

```env
DATABASE_URL=postgresql+psycopg://cybersentinel:cybersentinel@localhost:5432/cybersentinel
```

### Everything in containers

```bash
docker compose --profile full up -d
```

### With the fine-tuned model

```env
LLM_BACKEND=hf
MODEL_ADAPTER_PATH=models/cybersentinel-lora
```

`GET /health` reports which backend, vector store and database are actually
live, including whether any fallback is in use.

## 7. Dataset

A synthetic cybersecurity instruction corpus generated from hand-written event
templates, covering all 15 taxonomy categories including `Unknown`.

```bash
python scripts/prepare_dataset.py --per-category 120
```

The methodology point that matters: **templates are split between train,
validation and test before any instance is generated**, so a phrasing seen in
training never appears in the test set. Deduplication runs test-first, so
collisions are removed from training rather than from an evaluation set.

A separate hand-authored **hard test set** contains paraphrased attacks with no
signature vocabulary, benign events full of alarming keywords, and inputs where
the right answer is `Unknown`.

Full methodology, class distribution, licensing and limitations:
[`docs/dataset.md`](docs/dataset.md).

## 8. Fine-tuning (QLoRA)

```bash
pip install -e ".[ml]"
python scripts/train.py --dry-run       # validate config and data, no GPU needed
python scripts/train.py                 # train
python scripts/train.py --low-memory    # smaller GPU
```

4-bit NF4 quantisation, frozen base weights, LoRA adapters (rank 16) on
attention and MLP projections. Only the adapter is saved. Every hyperparameter
is in `configs/training.yaml`.

Details and hardware profiles: [`docs/fine_tuning.md`](docs/fine_tuning.md).

## 9. RAG pipeline

```
load ──► clean ──► chunk (with provenance header) ──► embed ──► Qdrant
query ──► normalise ──► embed ──► search ──► rerank ──► context with citations
```

The grounding rule, enforced in code:

> A MITRE technique, CWE or CVE is reported **only** if it appears in the
> retrieved context. Anything else the model proposes is recorded in
> `rejected_claims` and shown in the report.

Knowledge base: 40 ATT&CK techniques, 16 CWEs and 10 guidance documents from
NIST, OWASP and CISA, each with its source and canonical URL.

Details: [`docs/rag.md`](docs/rag.md).

## 10. LangGraph workflow

Real conditional routing and a real interrupt, not a chain:

- benign and unknown events **skip** retrieval;
- multi-event submissions reach correlation with per-event detections intact;
- high-risk findings **pause** at a checkpoint and resume when an analyst
  decides;
- a rejected recommendation set **loops back** once for investigative
  alternatives.

```python
run = workflow.analyze(event)     # pauses if approval is required
run.awaiting_approval             # True; no report exists yet
resumed = workflow.submit_decision(run.thread_id, "APPROVED", decided_by="analyst")
```

Details: [`docs/langgraph.md`](docs/langgraph.md).

## 11. Agents

| Agent | Kind | Responsibility |
|---|---|---|
| Input classifier | deterministic | Format detection, event splitting, indicator extraction |
| Threat detector | LLM | Classification, evidence extraction, severity |
| Threat intelligence | RAG + LLM | Retrieval and grounding, claim rejection |
| Correlation | deterministic + LLM | Shared indicators, kill-chain ordering |
| Risk assessment | deterministic | Likelihood × impact with printed rationale |
| Response | LLM + filter | Defensive recommendations, offensive actions blocked |
| Report | LLM + assembly | Structured report and explainability answers |

## 12. API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Component status, including active fallbacks |
| `POST /analyze` | Analyse an event; may pause for approval |
| `POST /analyze/batch` | Analyse several independent events |
| `POST /approval/{incident_id}` | Submit APPROVED / REJECTED / ESCALATED |
| `GET /incidents` | List stored incidents |
| `GET /incidents/{incident_id}` | Fetch one report |
| `GET /incidents/pending-approval` | Incidents awaiting a decision |
| `GET /threat-intelligence/search` | Search the knowledge base |
| `GET /indicators/search` | "Has this IP appeared before?" |
| `GET /metrics` | Dashboard aggregates |

Interactive documentation at `/docs`.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "47 failed SSH login attempts from 198.51.100.23 within 3 minutes."}'
```

## 13. Evaluation

```bash
python scripts/evaluate.py
python scripts/evaluate.py --with-hf --adapter models/cybersentinel-lora
```

Current results with the rules baseline:

| Study | Result |
|---|---|
| Detection (generated test set) | accuracy 1.00, macro F1 1.00, severity 0.67 |
| Detection (hard test set) | accuracy 0.11, macro F1 0.01 |
| RAG grounding | grounded rate 0.00 → **1.00**, MITRE recall 0.00 → **1.00** |
| Ablation | each layer's contribution isolated; workflow overhead ≈ 11 ms |
| Hallucination | insufficient-evidence rate **1.00**, asserted claims **0.00** |
| Retrieval | precision@5 0.77, recall@5 0.96, MRR 1.00 |
| Agent workflow | routing 1.00, approval gate 1.00, completion 1.00 |

The gap between 1.00 and 0.11 is deliberate and reported: the generated test set
is drawn from the same grammar as the training data, so a keyword system
saturates it. **The hard set is the meaningful comparison.**

Methodology, threats to validity and the research-question mapping:
[`docs/evaluation.md`](docs/evaluation.md).

## 14. Demo scenarios

Available in the dashboard's scenario picker:

| # | Scenario | Demonstrates |
|---|---|---|
| 1 | Phishing email | Email routing, T1566 grounding, credential-theft response |
| 2 | Brute force | T1110 mapping, HIGH risk, approval checkpoint |
| 3 | SQL injection | Payload analysis, CWE-89, parameterisation guidance |
| 4 | Multi-stage intrusion | Correlation → Reconnaissance → Credential Access → Privilege Escalation, CRITICAL |
| 5 | Benign activity | No forced classification, LOW risk, no approval |
| 6 | Insufficient evidence | Returns `Unknown` instead of inventing a threat |

Scenarios 5 and 6 are the important ones: they show the system declining to
invent a finding.

## 15. Testing

```bash
pytest                      # 227 tests
pytest tests/test_security.py -v
```

| File | Covers |
|---|---|
| `test_agents.py` | Taxonomy, classification, detection, risk, correlation, grounding, response, report |
| `test_graph.py` | Routing functions, state, approval flow, bounded loops, node failure isolation |
| `test_rag.py` | Chunking, embeddings, vector store, retrieval, grounding filters |
| `test_api.py` | Endpoints, validation, approval flow, persistence, incident memory |
| `test_security.py` | No code execution, no outbound fetches, no offensive recommendations, secret redaction |
| `test_ui.py` | Every dashboard page renders; the approval checkpoint appears |

Tests run with the mock backend, a local vector store and temporary SQLite: no
GPU, no network, no Docker.

## 16. Project structure

```
├── app/streamlit_app.py            SOC dashboard
├── configs/                        model.yaml, training.yaml, rag.yaml
├── data/
│   ├── knowledge_base/             curated ATT&CK / CWE / guidance corpus
│   ├── train|validation|test/      generated splits
│   └── test/hard_test.jsonl        hand-authored hard cases
├── docs/                           architecture, dataset, fine_tuning, rag,
│                                   langgraph, evaluation, security
├── evaluation/results/             evaluation output
├── scripts/                        prepare_dataset, train, evaluate, ingest
├── src/cybersentinel/
│   ├── agents/                     one module per agent
│   ├── api/                        FastAPI app, routes, schemas
│   ├── cybersecurity/              taxonomy, risk, mitre, correlation
│   ├── database/                   models, connection, repository
│   ├── evaluation/                 metrics, experiments, runner
│   ├── graph/                      state, nodes, edges, workflow
│   ├── llm/                        backends, prompts, structured output
│   ├── rag/                        loaders, chunking, embeddings, store
│   ├── schemas/                    Pydantic domain models
│   ├── training/                   templates, dataset, config, trainer
│   └── utils/                      config, logging, validation
└── tests/
```

## 17. Security considerations

- **Nothing is executed.** No `eval`, `exec`, `subprocess` or shell invocation
  exists in the package - a test scans the source to prove it.
- **Nothing is fetched.** Analyst-supplied URLs are analysed as text; a test
  monkeypatches `httpx` to fail on any outbound call and runs a URL through the
  full workflow.
- **No offensive recommendations.** Hacking back, scanning a source, deleting
  logs, disabling auditing and paying a ransom are filtered out even when the
  model proposes them.
- **Disruptive actions need a human.** Block, isolate, disable, revoke and
  similar always require approval, at any risk level.
- **Secrets are redacted** before logging, and only a redacted 500-character
  preview of the input is persisted.

Known gaps - no authentication, development database credentials, prompt
injection not fully solved - are documented in
[`docs/security.md`](docs/security.md).

## 18. Limitations

- The default backend is a deterministic rules analyst, not an LLM. Set
  `LLM_BACKEND=hf` for model-based analysis.
- The training corpus is synthetic and more regular than production telemetry.
- The generated test set is saturated by a keyword baseline; quote the hard set.
- The knowledge base is a curated subset, not a live threat-intelligence feed.
- Correlation produces a hypothesis from shared indicators and ordering, not
  proof of a single intrusion.
- The API has no authentication and is not deployment-ready as-is.
- English only.

The system does not claim to be accurate, autonomous or complete. Every finding
requires analyst validation.

## 19. Future work

- Multimodal screenshot analysis for suspicious emails
- Graph-based attack-chain visualisation
- Cross-encoder reranking
- Live CVE/NVD ingestion with scheduled refresh
- Analyst feedback loop feeding continual fine-tuning
- LLM-as-a-judge metrics alongside the deterministic ones
- Optional SIEM integration and MCP-based tool access
- Authentication, rate limiting and multi-tenancy for real deployment

---

## Licence

MIT. See [`LICENSE`](LICENSE).

MITRE ATT&CK® and CWE™ are trademarks of The MITRE Corporation. This project is
not affiliated with or endorsed by MITRE, NIST, OWASP or CISA. Knowledge-base
entries are summaries written for this project, with canonical URLs preserved so
every claim can be verified at its source.
