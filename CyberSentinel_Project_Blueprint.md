# CyberSentinel — Agentic Cybersecurity Intelligence Platform

## 1. Project Objective

Build a production-quality, academically defensible cybersecurity LLM project that combines:

- Fine-tuning of a pretrained model for Cybersecurity
- Cybersecurity threat detection and incident analysis
- Retrieval-Augmented Generation (RAG)
- MITRE ATT&CK / CVE / CWE threat intelligence
- LangGraph-based multi-agent orchestration
- Conditional routing and agent state
- Human-in-the-loop approval for high-impact response recommendations
- Incident correlation and memory
- Explainable security analysis
- Structured JSON outputs
- Comprehensive evaluation and ablation studies
- A simple SOC-style Streamlit dashboard
- FastAPI backend

The system should be designed as a college MSc-level LLM project. It must demonstrate real understanding of LLM fine-tuning, RAG, agents, LangGraph, cybersecurity, evaluation, and system architecture rather than being a superficial chatbot.

---

# 2. Project Title

Preferred title:

**CyberSentinel: An Agentic LLM Framework for Cybersecurity Threat Detection, Intelligence Retrieval and Incident Response**

Alternative:

**CyberAgent: A Fine-Tuned and RAG-Augmented LLM with LangGraph-Based Orchestration for Automated Cybersecurity Incident Analysis**

Use `CyberSentinel` as the project/application name throughout the codebase.

---

# 3. Core Problem Statement

Security analysts receive heterogeneous security events such as:

- suspicious emails
- phishing messages
- URLs
- authentication logs
- network alerts
- malware descriptions
- vulnerability descriptions
- SQL injection attempts
- XSS payloads
- brute-force activity
- suspicious user behavior

Traditional rule-based systems can detect known patterns but struggle with natural-language interpretation, contextual reasoning, cross-source correlation, and explainable recommendations.

CyberSentinel should provide an AI-assisted security analysis pipeline that:

1. Accepts a cybersecurity event.
2. Determines the event category.
3. Detects the likely attack/threat type using a cybersecurity-specialized fine-tuned LLM.
4. Retrieves relevant authoritative cybersecurity knowledge.
5. Maps findings to MITRE ATT&CK / CVE / CWE where appropriate.
6. Correlates related alerts when multiple events are supplied.
7. Calculates a risk/severity assessment.
8. Generates recommended mitigation/response actions.
9. Requests human approval before high-impact response recommendations.
10. Produces a structured and explainable incident report.

The system should be recommendation-oriented. It must NOT autonomously execute destructive or intrusive security actions.

---

# 4. High-Level Architecture

Implement this conceptual architecture:

User / SOC Analyst
        |
        v
Input Interface
        |
        v
FastAPI API
        |
        v
LangGraph Orchestrator
        |
        +-----------------------+
        |                       |
        v                       v
Input Classification       Incident Context
        |                       |
        +-----------+-----------+
                    |
                    v
          Threat Detection Agent
                    |
                    v
          Fine-Tuned Cyber LLM
                    |
                    v
           Threat Intelligence
                    |
                    v
                  RAG
                    |
        +-----------+-----------+
        |                       |
        v                       v
MITRE ATT&CK / CVE/CWE     Security Documents
        |                       |
        +-----------+-----------+
                    |
                    v
          Correlation Agent
                    |
                    v
          Risk Assessment Agent
                    |
                    v
            Human Approval
                    |
          +---------+---------+
          |                   |
       Approved             Rejected
          |                   |
          v                   v
 Response Recommendation   Re-analysis /
          |                Escalation
          +---------+---------+
                    |
                    v
          Incident Report Agent
                    |
                    v
              Final Output
                    |
                    v
             Streamlit SOC UI

---

# 5. Recommended Technology Stack

## LLM

Primary recommendation:

- Qwen2.5-3B-Instruct

The implementation should make the base model configurable through environment variables so another Hugging Face instruction model can be substituted later.

Example:

`BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct`

If the exact model identifier has changed, verify the current Hugging Face repository before implementation rather than inventing a model ID.

## Fine-Tuning

Use:

- Hugging Face Transformers
- PEFT
- TRL
- bitsandbytes
- PyTorch
- Accelerate

Use QLoRA where GPU memory makes it appropriate.

## Agent Orchestration

- LangGraph
- LangChain where useful

Do not use LangGraph merely as a linear chain. Use actual stateful routing and conditional edges.

## RAG

- Qdrant
- Sentence Transformers or another suitable embedding model
- LangChain/LangGraph-compatible retrieval utilities where useful

## Backend

- FastAPI
- Pydantic

## Database

Use PostgreSQL for persistent incident/application metadata where useful.

Keep the RAG vector store in Qdrant.

## UI

Use Streamlit for the first implementation.

The UI should prioritize functionality and clarity over visual complexity.

## Development

- Python 3.11+
- Git
- `.env`
- `.env.example`
- `requirements.txt` or preferably `pyproject.toml`
- Docker/Docker Compose where practical

---

# 6. Functional Requirements

## 6.1 Input Types

The application should support at least:

### A. Free-text security alert

Example:

"47 failed SSH login attempts from the same external IP within 3 minutes."

### B. Log input

Allow users to paste/upload logs.

### C. Suspicious email

Allow:

- sender
- subject
- body
- URLs

### D. URL

Analyze the URL textually/reputationally where supported by the project.

Do not build an unsafe live malware crawler.

### E. Multiple events

Allow multiple related alerts to be supplied for correlation.

---

# 7. Cybersecurity Categories

The initial supported threat categories should include:

1. Phishing
2. Brute Force
3. Credential Attacks
4. SQL Injection
5. Cross-Site Scripting (XSS)
6. Malware
7. DDoS / Network Attack
8. Port Scanning / Reconnaissance
9. Privilege Escalation
10. Data Exfiltration
11. Suspicious Authentication Activity
12. Vulnerability / Exploit-related event
13. Insider-threat-like anomalous behavior
14. Benign / Unknown

The system must support an `Unknown` / `Insufficient Evidence` result rather than forcing every input into a known attack class.

---

# 8. Fine-Tuned Model Responsibilities

The fine-tuned model is the cybersecurity-domain analysis component.

It should learn tasks such as:

- threat classification
- severity estimation
- cybersecurity reasoning
- evidence extraction
- MITRE technique suggestion
- mitigation recommendation
- structured incident analysis

The model should NOT be responsible for current threat intelligence retrieval. That is the job of RAG.

This separation must be clearly documented.

---

# 9. Fine-Tuning Dataset

Create a reproducible cybersecurity instruction dataset.

Do not blindly download a random dataset and call it fine-tuning.

Create a normalized training schema.

Suggested format:

```json
{
  "instruction": "Analyze the following cybersecurity event.",
  "input": "Multiple failed SSH logins from the same IP...",
  "output": {
    "attack_type": "Brute Force",
    "severity": "HIGH",
    "confidence": 0.94,
    "evidence": [
      "Repeated failed authentication attempts",
      "Same source IP",
      "Short time window"
    ],
    "mitre_technique": "T1110",
    "recommendations": [
      "Investigate source IP",
      "Enable authentication rate limiting",
      "Review successful logins"
    ]
  }
}
```

Use consistent JSON output during training where practical.

---

# 10. Dataset Categories

Construct/normalize examples for:

- Phishing
- Brute force
- SQL injection
- XSS
- Malware
- DDoS
- Port scanning
- Credential attacks
- Privilege escalation
- Data exfiltration
- Vulnerabilities
- Authentication anomalies
- Benign/normal activity

Include diverse examples and paraphrases.

Avoid training-test contamination.

Do not use test examples to generate training examples.

---

# 11. Dataset Splitting

Use:

- 80% training
- 10% validation
- 10% held-out test

Use stratification where appropriate.

Document:

- dataset sources
- licenses
- preprocessing
- class distribution
- deduplication
- train/validation/test methodology

If synthetic examples are generated, clearly label them as synthetic and document how they were created.

---

# 12. QLoRA Fine-Tuning

Implement a reproducible training script.

Expected conceptual flow:

Base pretrained model
        |
        v
4-bit quantization
        |
        v
Freeze base model
        |
        v
Attach LoRA adapters
        |
        v
Train adapters
        |
        v
Save adapter
        |
        v
Evaluation

The training configuration should expose:

- model name
- dataset path
- output directory
- learning rate
- batch size
- gradient accumulation
- number of epochs
- max sequence length
- LoRA rank
- LoRA alpha
- LoRA dropout
- quantization settings

Do not hardcode machine-specific paths.

---

# 13. Training Hardware

The implementation should be designed to run on a consumer NVIDIA GPU where feasible.

Provide:

1. GPU training configuration.
2. Lower-memory fallback configuration.
3. CPU inference support where practical.

Do not assume an unlimited GPU.

Use:

- gradient accumulation
- mixed precision
- 4-bit quantization
- gradient checkpointing where useful

---

# 14. Fine-Tuning Evaluation

Evaluate the base model and fine-tuned model on the same held-out test set.

Measure at least:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Per-class metrics
- Confusion matrix

For structured generation, also measure:

- JSON validity
- field completeness
- attack classification correctness
- severity correctness
- MITRE mapping accuracy where applicable

The report should explicitly compare:

**Base model vs Fine-tuned model**

---

# 15. RAG Knowledge Base

Build a cybersecurity knowledge base using authoritative/public sources.

Prioritize:

- MITRE ATT&CK
- CVE/NVD information
- CWE
- reputable security documentation
- official cybersecurity guidance

Do not rely exclusively on random blogs.

The ingestion pipeline should:

1. Load documents.
2. Clean text.
3. Split/chunk documents.
4. Generate embeddings.
5. Store vectors in Qdrant.
6. Preserve metadata.

Metadata should include where possible:

```json
{
  "source": "MITRE ATT&CK",
  "document_id": "T1110",
  "title": "Brute Force",
  "url": "...",
  "category": "attack-pattern"
}
```

Never fabricate citations or source URLs.

---

# 16. RAG Pipeline

Implement:

Query
  |
  v
Query normalization
  |
  v
Embedding
  |
  v
Qdrant similarity search
  |
  v
Top-k candidates
  |
  v
Optional reranking
  |
  v
Relevant context
  |
  v
LLM reasoning
  |
  v
Grounded response

The system should return source metadata for retrieved information.

---

# 17. RAG Safety / Grounding

The system must distinguish:

- model inference
- retrieved fact
- analyst recommendation

For example:

```json
{
  "finding": "Likely brute force attack",
  "evidence": ["47 failed login attempts"],
  "threat_intelligence": [
    {
      "source": "MITRE ATT&CK",
      "technique": "T1110"
    }
  ],
  "recommendation": "Investigate and rate-limit repeated authentication attempts"
}
```

The LLM must not invent CVE IDs or MITRE techniques.

If evidence is insufficient, return:

`Insufficient evidence`

rather than hallucinating.

---

# 18. LangGraph Architecture

Use LangGraph as the central orchestration layer.

Create a typed state object.

Suggested state:

```python
class CyberState(TypedDict):
    input_text: str
    input_type: str
    events: list
    classification: dict
    threat_analysis: dict
    retrieved_context: list
    mitre_mapping: dict
    correlated_incidents: list
    risk_assessment: dict
    response_recommendations: list
    human_approval: str
    final_report: dict
    messages: list
    errors: list
```

Use Pydantic models for externally exposed structured data.

---

# 19. LangGraph Nodes

Implement at least these nodes:

## Node 1 — Input Classifier

Determine whether the input is:

- phishing
- log
- URL
- vulnerability
- network event
- general security alert
- multiple-event incident

Return structured classification.

---

## Node 2 — Threat Detection Agent

Call the fine-tuned cybersecurity model.

Return:

```json
{
  "attack_type": "...",
  "confidence": 0.0,
  "severity": "...",
  "evidence": []
}
```

---

## Node 3 — Threat Intelligence / RAG Agent

Use the detection result to retrieve:

- MITRE ATT&CK
- CVE/CWE
- security guidance

Return grounded sources.

---

## Node 4 — Correlation Agent

If multiple events are present, identify relationships.

Example:

Port scan
  ->
Brute force
  ->
Successful login
  ->
Privilege escalation

Return:

- possible attack chain
- related events
- confidence
- supporting evidence

Do not claim a confirmed attack chain without sufficient evidence.

---

## Node 5 — Risk Assessment Agent

Calculate risk using a transparent framework.

Example:

Likelihood: 1-5
Impact: 1-5

Risk Score:

`Likelihood * Impact`

Map to:

- LOW
- MEDIUM
- HIGH
- CRITICAL

Document the mapping.

The risk score must not be arbitrary or hidden.

---

## Node 6 — Human Approval

For high/critical severity or high-impact response recommendations, require analyst review.

Example:

"Recommended action: block source IP."

Show:

- evidence
- confidence
- retrieved sources
- recommendation

Then wait for:

- APPROVE
- REJECT
- ESCALATE

Do not automatically execute the action.

---

## Node 7 — Response Recommendation Agent

Generate defensive recommendations.

Examples:

- investigate affected account
- rotate credentials
- enable MFA
- rate-limit authentication
- inspect related logs
- patch vulnerable component
- isolate affected host
- preserve forensic evidence

Avoid destructive automation.

---

## Node 8 — Incident Report Agent

Generate a structured report:

```json
{
  "incident_id": "...",
  "summary": "...",
  "attack_type": "...",
  "severity": "...",
  "confidence": 0.0,
  "evidence": [],
  "mitre_techniques": [],
  "cve_cwe": [],
  "correlated_events": [],
  "risk_score": 0,
  "recommendations": [],
  "sources": []
}
```

---

# 20. Conditional LangGraph Routing

LangGraph should contain actual conditional routing.

Example:

START
  |
Input Classifier
  |
  +-- phishing --> Phishing Analysis
  |
  +-- URL ------> URL Analysis
  |
  +-- log ------> Log Analysis
  |
  +-- vulnerability --> Vulnerability Analysis
  |
  +-- multi-event --> Correlation
  |
  v
Threat Detection
  |
RAG
  |
Risk Assessment
  |
  +-- HIGH/CRITICAL --> Human Approval
  |
  +-- LOW/MEDIUM ----> Response Recommendation
  |
Human Approval
  |
  +-- APPROVE --> Response Recommendation
  |
  +-- REJECT --> Re-analysis / Escalation
  |
  v
Report
  |
 END

---

# 21. Memory and Persistence

Implement incident history.

Store:

- incident ID
- timestamp
- input
- detected threat
- severity
- risk score
- MITRE mapping
- sources
- recommendations
- approval status

Use PostgreSQL for durable metadata.

The system should be able to answer:

"Has this IP appeared in previous incidents?"

"Have we seen this attack pattern before?"

Do not expose sensitive personal data unnecessarily.

---

# 22. Incident Correlation

This is a key differentiating feature.

Example input:

```text
Event 1:
Port scan from IP X

Event 2:
20 failed SSH logins from IP X

Event 3:
Successful SSH login from IP X

Event 4:
Privilege escalation shortly after login
```

The correlation agent should identify:

```text
Possible multi-stage intrusion

Reconnaissance
    ->
Credential Access
    ->
Initial Access
    ->
Privilege Escalation
```

It should map techniques to MITRE ATT&CK only when supported by retrieved evidence.

---

# 23. Explainability

Every final result should explain:

1. What was detected?
2. Why was it detected?
3. What evidence supports it?
4. What is the confidence?
5. Which threat intelligence sources support it?
6. Which MITRE technique is relevant?
7. What is the risk?
8. What should the analyst do next?

Never return only:

`Attack = Phishing`

---

# 24. Streamlit SOC Dashboard

Build a simple dashboard.

Pages/sections:

## Dashboard

Show:

- total incidents
- critical incidents
- high incidents
- attack distribution
- recent incidents

## Analyze Incident

Input:

- alert text
- logs
- email
- URL
- multiple events

Display:

- threat type
- confidence
- severity
- evidence

## Threat Intelligence

Display:

- MITRE technique
- CVE/CWE
- retrieved documents
- source metadata

## Attack Chain

Visualize:

Recon
  ->
Initial Access
  ->
Execution
  ->
Privilege Escalation
  ->
Exfiltration

Only show stages supported by evidence.

## Incident Report

Show structured report.

## Human Approval

Display pending high-risk recommendations.

---

# 25. FastAPI Endpoints

Create clean APIs.

Suggested endpoints:

`GET /health`

`POST /analyze`

`POST /analyze/batch`

`GET /incidents`

`GET /incidents/{incident_id}`

`GET /threat-intelligence/search`

`POST /approval/{incident_id}`

`GET /metrics`

Use Pydantic request/response schemas.

Include proper error handling.

---

# 26. Security Requirements

Because this is a cybersecurity application, follow secure development practices.

- Never execute uploaded shell commands.
- Never execute arbitrary user-provided code.
- Never automatically launch penetration tests.
- Never perform destructive actions.
- Sanitize inputs.
- Validate uploaded files.
- Restrict file sizes.
- Keep secrets in `.env`.
- Never commit API keys.
- Add `.env` to `.gitignore`.
- Use safe logging.
- Avoid storing unnecessary sensitive data.
- Clearly label recommendations as recommendations.

This project is a defensive analysis system, not an offensive security automation tool.

---

# 27. Evaluation Framework

The project must include a serious evaluation section.

## Experiment 1 — Base vs Fine-Tuned

Compare:

- base model
- fine-tuned model

Metrics:

- accuracy
- precision
- recall
- macro F1
- JSON validity

---

## Experiment 2 — Fine-Tuned vs Fine-Tuned + RAG

Evaluate:

- factual correctness
- MITRE mapping
- CVE/CWE grounding
- hallucination rate
- citation/source accuracy

---

## Experiment 3 — Pipeline Ablation

Compare:

1. Base LLM
2. Fine-tuned LLM
3. Fine-tuned + RAG
4. Fine-tuned + RAG + LangGraph

Measure:

- classification quality
- groundedness
- response quality
- latency
- token usage
- failure rate

---

# 28. Hallucination Evaluation

Create adversarial/unknown examples.

Example:

Input contains insufficient evidence.

Expected behavior:

`Insufficient evidence`

rather than invented:

`CVE-XXXX-XXXX`

Measure:

- hallucination rate
- unsupported MITRE mapping rate
- unsupported CVE rate
- grounded answer rate

---

# 29. RAG Evaluation

Measure at least:

- retrieval precision
- retrieval recall where ground truth exists
- source relevance
- answer groundedness
- citation correctness

If advanced evaluation tooling is used, keep it optional and make sure the core evaluation can run without expensive external APIs.

---

# 30. Agent Evaluation

Measure:

- correct routing
- tool selection
- successful workflow completion
- structured output validity
- human-approval behavior
- failure recovery
- unsupported-claim rate

Log LangGraph transitions for debugging.

---

# 31. Performance Evaluation

Measure:

- end-to-end latency
- LLM inference latency
- RAG latency
- number of retrieved documents
- token usage
- memory usage where practical

Do not optimize prematurely.

---

# 32. Observability

Add structured logs.

Each run should have:

- run ID
- incident ID
- node name
- timestamp
- status
- latency
- errors

Example:

```text
[RUN] 123
[Node] input_classifier
[Status] success
[Latency] 0.81s
```

Avoid logging secrets or sensitive raw inputs unnecessarily.

---

# 33. Error Handling

Every agent should gracefully handle:

- model failure
- invalid JSON
- missing RAG results
- Qdrant unavailable
- database unavailable
- malformed input
- empty input
- timeout

Use retries only where safe.

If the model produces invalid JSON:

1. attempt structured parsing
2. optionally run a controlled repair step
3. if still invalid, return a safe error

Do not silently fabricate missing values.

---

# 34. Project Folder Structure

Use a clean modular architecture:

```text
cybersentinel/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── docker-compose.yml
│
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   └── rag.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ingest_knowledge_base.py
│   └── build_embeddings.py
│
├── src/
│   └── cybersentinel/
│       ├── __init__.py
│       │
│       ├── api/
│       │   ├── main.py
│       │   ├── routes/
│       │   └── schemas/
│       │
│       ├── agents/
│       │   ├── input_classifier.py
│       │   ├── threat_detector.py
│       │   ├── threat_intelligence.py
│       │   ├── correlation.py
│       │   ├── risk_assessment.py
│       │   ├── response.py
│       │   └── report.py
│       │
│       ├── graph/
│       │   ├── state.py
│       │   ├── nodes.py
│       │   ├── edges.py
│       │   └── workflow.py
│       │
│       ├── llm/
│       │   ├── model.py
│       │   ├── inference.py
│       │   ├── prompts.py
│       │   └── structured_output.py
│       │
│       ├── training/
│       │   ├── dataset.py
│       │   ├── config.py
│       │   └── trainer.py
│       │
│       ├── rag/
│       │   ├── loaders.py
│       │   ├── chunking.py
│       │   ├── embeddings.py
│       │   ├── vectorstore.py
│       │   └── retriever.py
│       │
│       ├── cybersecurity/
│       │   ├── mitre.py
│       │   ├── risk.py
│       │   ├── taxonomy.py
│       │   └── correlation.py
│       │
│       ├── database/
│       │   ├── models.py
│       │   ├── repository.py
│       │   └── connection.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── validation.py
│           └── config.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_agents.py
│   ├── test_graph.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_security.py
│
├── evaluation/
│   ├── results/
│   ├── plots/
│   └── reports/
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── fine_tuning.md
    ├── rag.md
    ├── langgraph.md
    ├── evaluation.md
    └── security.md
```

---

# 35. Environment Variables

Create `.env.example`.

Potential variables:

```env
BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

HF_TOKEN=

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

DATABASE_URL=

EMBEDDING_MODEL_NAME=

LOG_LEVEL=INFO

MODEL_ADAPTER_PATH=

TOP_K=5
```

Never place actual secrets in source code.

---

# 36. Docker

Provide Docker Compose for:

- Qdrant
- PostgreSQL
- optionally FastAPI

The LLM training process does not need to run inside Docker initially.

Document both:

- local development
- Docker-based services

---

# 37. Prompt Engineering

Create centralized prompt templates.

Do not scatter prompts throughout the code.

Prompts should explicitly instruct the model:

- use only provided evidence
- do not invent CVEs
- do not invent MITRE techniques
- state uncertainty
- return structured output
- distinguish evidence from recommendation

---

# 38. Structured Output

Use Pydantic models such as:

```python
class ThreatAnalysis(BaseModel):
    attack_type: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
    confidence: float
    evidence: list[str]
    mitre_techniques: list[str]
    reasoning: str
```

Use validation constraints:

- confidence between 0 and 1
- severity from fixed values
- non-empty attack type
- evidence as list

---

# 39. Risk Model

Implement a transparent risk matrix.

Example:

Likelihood:

1 = Rare
2 = Unlikely
3 = Possible
4 = Likely
5 = Almost Certain

Impact:

1 = Negligible
2 = Minor
3 = Moderate
4 = Major
5 = Severe

Risk:

`Risk Score = Likelihood × Impact`

Example mapping:

1-4 = LOW
5-9 = MEDIUM
10-16 = HIGH
17-25 = CRITICAL

Document that this is a project risk model, not a universal cybersecurity standard.

---

# 40. Human-in-the-Loop Requirements

Human approval should be triggered for:

- CRITICAL incidents
- HIGH-risk actions
- actions that could block accounts/IPs
- actions that could isolate hosts
- other operationally impactful recommendations

The UI should display:

- recommendation
- evidence
- confidence
- risk
- sources

Then require explicit approval.

---

# 41. Demo Scenarios

Prepare at least 5 polished demo cases.

## Demo 1 — Phishing

Input:

A suspicious email containing an urgent account-verification request and suspicious URL.

Expected:

- phishing classification
- evidence
- severity
- threat intelligence
- recommendation

## Demo 2 — Brute Force

Input:

Repeated failed SSH authentication.

Expected:

- brute-force detection
- T1110 mapping
- high severity
- response recommendation

## Demo 3 — SQL Injection

Input:

A suspicious SQL query payload.

Expected:

- SQL injection
- relevant CWE where supported
- explanation
- mitigation

## Demo 4 — Multi-stage attack

Input:

Port scan + failed login + successful login + privilege escalation.

Expected:

- correlation
- possible attack chain
- multiple techniques
- high/critical risk
- human approval

## Demo 5 — Unknown / Benign

Input:

A normal authentication event.

Expected:

- benign/unknown
- low risk
- no forced attack classification

This demonstrates that the system does not hallucinate threats.

---

# 42. Academic Contribution

The final report should clearly state contributions:

1. Cybersecurity-specific LLM fine-tuning.
2. Parameter-efficient QLoRA adaptation.
3. Retrieval-grounded threat intelligence.
4. LangGraph-based multi-agent orchestration.
5. Conditional security workflow routing.
6. Incident correlation.
7. Human-in-the-loop cybersecurity response.
8. Explainable threat analysis.
9. Hallucination/grounding evaluation.
10. Ablation study showing the contribution of each component.

---

# 43. Research Questions

Use these in the report:

### RQ1

Does cybersecurity-specific QLoRA fine-tuning improve threat classification compared with the base instruction model?

### RQ2

Does RAG improve cybersecurity factual grounding and reduce unsupported threat-intelligence claims?

### RQ3

Does LangGraph-based orchestration improve multi-step incident analysis compared with a single LLM call?

### RQ4

Can incident correlation identify plausible multi-stage attack patterns from multiple security events?

### RQ5

How does human-in-the-loop approval improve safety for high-impact response recommendations?

---

# 44. Important Design Principle

Do not make every component an LLM.

Use deterministic code where deterministic logic is better.

Examples:

- risk score calculation -> Python
- input validation -> Pydantic
- confidence range validation -> Pydantic
- database persistence -> SQL
- vector retrieval -> Qdrant
- routing -> LangGraph conditional edges
- attack classification/reasoning -> LLM
- report generation -> LLM
- source retrieval -> RAG

This makes the architecture more reliable and easier to defend during a viva.

---

# 45. Important Design Principle: Fine-Tuning vs RAG

Document this distinction clearly.

Fine-tuning:

**Teaches the model cybersecurity behavior/patterns.**

RAG:

**Provides external/current knowledge at inference time.**

Do not use fine-tuning as a replacement for a continuously updated knowledge base.

Do not put every cybersecurity document into the model weights.

---

# 46. Important Design Principle: LangGraph vs LangChain

Use LangGraph for:

- workflow
- state
- routing
- cycles
- persistence
- human approval

Use LangChain components only where useful for:

- model wrappers
- retrievers
- document processing
- tool integration

Do not introduce unnecessary abstractions.

---

# 47. Testing Requirements

Write unit tests for:

- risk scoring
- classification parsing
- Pydantic validation
- RAG retrieval
- routing
- approval logic
- database repository
- API endpoints

Add integration tests for:

- full LangGraph workflow
- RAG + LLM
- FastAPI + workflow

Use mocked LLM responses in most tests.

Do not require an expensive GPU for ordinary unit tests.

---

# 48. Documentation Requirements

README must include:

1. Project overview
2. Problem statement
3. Architecture diagram
4. Technology stack
5. Dataset
6. Fine-tuning methodology
7. QLoRA explanation
8. RAG pipeline
9. LangGraph workflow
10. Agent descriptions
11. API documentation
12. Setup instructions
13. Training instructions
14. Evaluation instructions
15. Demo instructions
16. Limitations
17. Security considerations
18. Future work

Also create separate docs for:

- architecture
- fine-tuning
- RAG
- LangGraph
- evaluation
- security

---

# 49. Final Deliverables

The final project should contain:

- Working source code
- Dataset preparation pipeline
- Fine-tuning script
- Saved LoRA adapter
- Evaluation scripts
- RAG ingestion pipeline
- Qdrant collection setup
- LangGraph workflow
- FastAPI backend
- Streamlit dashboard
- PostgreSQL persistence
- Unit/integration tests
- Docker Compose
- `.env.example`
- README
- Architecture documentation
- Evaluation report
- Example demo inputs
- Evaluation results
- Architecture diagram
- LangGraph diagram

---

# 50. Development Strategy

Implement incrementally.

## Phase 1 — Project skeleton

Create:

- repository
- package structure
- configuration
- logging
- FastAPI health endpoint
- Streamlit shell

Do not implement everything at once.

## Phase 2 — Dataset

- identify datasets
- document licenses
- normalize
- clean
- deduplicate
- split
- create training format

## Phase 3 — Fine-Tuning

- implement QLoRA
- train
- save adapter
- evaluate
- compare with base model

## Phase 4 — RAG

- collect authoritative documents
- ingestion
- chunking
- embeddings
- Qdrant
- retrieval
- source metadata

## Phase 5 — LangGraph

Implement:

Input
 -> Classifier
 -> Threat Detector
 -> RAG
 -> Risk
 -> Response
 -> Report

Then add conditional routing.

## Phase 6 — Correlation + Memory

Add:

- multi-event correlation
- incident history
- previous incident lookup

## Phase 7 — Human Approval

Add approval checkpoint.

## Phase 8 — UI

Connect Streamlit to FastAPI.

## Phase 9 — Evaluation

Run:

- base vs fine-tuned
- fine-tuned vs RAG
- ablation
- hallucination
- retrieval
- agent workflow
- latency

## Phase 10 — Documentation

Finalize:

- README
- architecture
- report
- screenshots
- demo scenarios

---

# 51. Coding Standards

Follow:

- Python type hints
- Pydantic models
- modular functions
- clear docstrings
- meaningful variable names
- no giant files
- no hardcoded secrets
- no hardcoded absolute paths
- configuration through environment/config files
- structured logging
- testable functions
- clean exception handling

Use async FastAPI endpoints where appropriate.

Do not over-engineer.

---

# 52. Do Not Overclaim

The system should never claim:

- "100% accurate"
- "fully autonomous cybersecurity"
- "guaranteed attack detection"
- "guaranteed zero hallucinations"

Use language such as:

- "AI-assisted"
- "probable attack"
- "confidence"
- "evidence"
- "recommended response"
- "requires analyst validation"

---

# 53. Final Success Criteria

The project is considered complete only when:

- A cybersecurity event can be submitted.
- LangGraph routes the event.
- The fine-tuned model performs threat analysis.
- RAG retrieves relevant cybersecurity information.
- Sources are visible.
- MITRE/CVE/CWE mappings are grounded where applicable.
- Risk is calculated.
- High-impact recommendations trigger human approval.
- The final incident report is generated.
- Incident history is stored.
- Multiple alerts can be correlated.
- Base vs fine-tuned evaluation exists.
- Fine-tuned vs RAG evaluation exists.
- Ablation study exists.
- Hallucination/grounding evaluation exists.
- Tests pass.
- README explains how to run everything.
- The application can run without retraining the model.
- No secrets are committed.
- No destructive security actions are automatically executed.

---

# 54. Recommended Future Extensions

Only implement these after the core system works:

- multimodal suspicious-email screenshot analysis
- graph-based attack-chain visualization
- advanced reranking
- additional cybersecurity datasets
- continual fine-tuning
- model quantization for deployment
- more sophisticated agent memory
- analyst feedback loop
- evaluation using additional LLM-as-a-judge metrics
- optional SIEM integration
- MCP-based security tool integration

Do not let future extensions compromise the core project.

---

# 55. Final Architecture Summary

The final system should demonstrate the following relationship:

                    CYBERSENTINEL
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   Fine-Tuning          RAG          LangGraph
        |                |                |
     QLoRA          Qdrant + KB       Agents
        |                |                |
        +----------------+----------------+
                         |
                         v
                Cybersecurity Analysis
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Threat Detection       Incident Correlation
             |                       |
             +-----------+-----------+
                         |
                         v
                  Risk Assessment
                         |
                         v
                  Human Approval
                         |
                         v
               Response Recommendation
                         |
                         v
                  Incident Report

This separation of concerns is intentional:

- **Fine-tuning** = cybersecurity specialization
- **RAG** = grounded external knowledge
- **LangGraph** = orchestration and stateful workflow
- **Agents** = specialized reasoning tasks
- **Deterministic code** = validation, scoring, persistence
- **Human-in-the-loop** = safety
- **Evaluation** = scientific/academic validation

Build the project with this architecture as the target, but prioritize a working vertical slice early instead of implementing every advanced feature before the core workflow works.
