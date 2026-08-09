# UI specification

A description of the CyberSentinel dashboard as built, written so someone can
rebuild it in a different framework without reading the Python.

Current implementation: `app/streamlit_app.py` (Streamlit, ~600 lines, single
file). It is deliberately plain — the priority was that every number on screen
can be traced to how it was produced. A replacement is free to look completely
different, provided it keeps the data contracts and the safety rules below.

---

## 1. What the product is

An analyst console for AI-assisted security incident analysis. A user pastes in a
security event; the system classifies it, grounds it in threat intelligence,
scores its risk, and recommends defensive actions. Anything operationally
disruptive pauses for explicit human approval.

**Primary user:** a SOC analyst triaging alerts. Assume technical literacy,
assume time pressure, assume they must justify decisions to someone else later.

**Core promise the UI must convey:** this is a recommendation engine, not an
automation engine. It never acts.

---

## 2. Backend contract

The UI is a pure client over HTTP. Base URL default `http://localhost:8000`.
Interactive schema at `/docs`, machine-readable at `/openapi.json`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Component status; drives a status indicator |
| POST | `/analyze` | Analyse one event. May return a paused run |
| POST | `/analyze/batch` | Analyse several independent events |
| POST | `/approval/{incident_id}` | Submit the analyst decision, resumes the run |
| GET | `/incidents` | List stored incidents (filters: `severity`, `attack_type`, `approval_status`, `limit`, `offset`) |
| GET | `/incidents/{incident_id}` | One stored report |
| GET | `/incidents/pending-approval` | Incidents awaiting a decision |
| GET | `/threat-intelligence/search` | Search the knowledge base (`q`, `top_k`) |
| GET | `/indicators/search` | Previous incidents for an indicator (`value`, `kind`) |
| GET | `/metrics` | Dashboard aggregates (`days` optional) |

### `POST /analyze` request

```json
{
  "text": "47 failed SSH login attempts from 198.51.100.23 within 3 minutes.",
  "use_rag": true,
  "use_llm_response": true,
  "asset_criticality": 3
}
```

`text` is 1-20,000 characters. `asset_criticality` is 1-5 or null.

### `POST /analyze` response

```json
{
  "incident_id": "INC-20260809-C322C9",
  "run_id": "6aa712c059a0",
  "thread_id": "INC-20260809-C322C9",
  "awaiting_approval": true,
  "report": null,
  "pending_approval": { },
  "history_matches": [],
  "node_path": ["input_classifier", "threat_detector", "..."],
  "metrics": {},
  "errors": []
}
```

**The critical branch:** `awaiting_approval` decides what renders.

- `true` → `report` is **null**, `pending_approval` is populated. Render the
  approval checkpoint. There is deliberately no report yet: the workflow is
  paused mid-execution and no conclusion has been finalised.
- `false` → `pending_approval` is **null**, `report` is populated. Render the
  full report.

### `pending_approval` shape

```json
{
  "reason": "Risk level HIGH is at or above the HIGH threshold.",
  "attack_type": "Brute Force",
  "severity": "HIGH",
  "confidence": 0.85,
  "risk_level": "HIGH",
  "risk_score": 15,
  "evidence": ["47 failed authentication attempts in a 3 minute window", "..."],
  "recommendations": [
    {
      "action": "Review authentication logs for the source IP",
      "rationale": "Distinguishes a failed campaign from an actual compromise.",
      "priority": "HIGH",
      "high_impact": false,
      "requires_approval": false
    }
  ],
  "high_impact_actions": ["Block the source IP at the perimeter firewall"],
  "sources": [ ],
  "mitre_techniques": ["T1110", "T1110.001", "T1110.003"]
}
```

### `report` shape

```json
{
  "incident_id": "INC-...",
  "created_at": "2026-08-09T09:36:35Z",
  "summary": "Activity consistent with a probable Brute Force ...",
  "input_type": "alert",
  "attack_type": "Brute Force",
  "severity": "HIGH",
  "confidence": 0.85,
  "evidence": ["..."],
  "reasoning": "...",

  "mitre": {
    "techniques": [
      {"technique_id": "T1110", "name": "Brute Force", "tactic": "Credential Access",
       "tactic_id": "TA0006", "url": "https://attack.mitre.org/techniques/T1110/",
       "source": "MITRE ATT&CK"}
    ],
    "tactics": ["Credential Access"],
    "cwe": [{"cwe_id": "CWE-307", "name": "...", "url": "...", "source": "CWE"}],
    "cve": [{"cve_id": "CVE-...", "url": "...", "source": "NVD"}],
    "rejected_claims": ["T9999"],
    "grounded": true
  },

  "correlation": {
    "is_correlated": true,
    "confidence": 0.75,
    "shared_indicators": {"ips": ["203.0.113.45"]},
    "attack_chain": [
      {"stage": "Reconnaissance", "tactic_id": "TA0043", "event_indices": [0],
       "description": "...", "supporting_evidence": ["..."]}
    ],
    "summary": "...",
    "caveat": "Chain is a hypothesis ... requires analyst validation."
  },

  "risk": {
    "likelihood": 5, "impact": 3, "risk_score": 15, "risk_level": "HIGH",
    "likelihood_label": "Almost Certain", "impact_label": "Moderate",
    "formula": "risk_score = likelihood x impact",
    "rationale": ["Base likelihood 5 from detection confidence >= 0.90.", "..."]
  },

  "recommendations": [ ],
  "approval": {
    "decision": "APPROVED", "required": true, "reason": "...",
    "decided_by": "analyst", "decided_at": "...", "note": null
  },
  "sources": [
    {"source": "MITRE ATT&CK", "document_id": "T1110", "title": "Brute Force",
     "url": "https://...", "category": "attack-pattern"}
  ],
  "explainability": {
    "what_was_detected": "...", "why_detected": "...", "evidence": "...",
    "confidence": "...", "threat_intelligence_sources": "...",
    "mitre_techniques": "...", "risk": "...", "next_steps": "..."
  },
  "errors": [],
  "latency_seconds": 0.31,
  "disclaimer": "AI-assisted analysis. Findings are probabilistic and require analyst validation. No response action is executed automatically."
}
```

### `POST /approval/{incident_id}` request

```json
{"decision": "APPROVED", "decided_by": "analyst-name", "note": "optional"}
```

`decision` is one of `APPROVED`, `REJECTED`, `ESCALATED`. The path parameter is
the `thread_id` from the analyse response. Response is the same
`AnalyzeResponse` shape, now with `awaiting_approval: false` and a populated
`report`.

Errors: `409` if no run is paused for that id, `422` for an invalid decision.

### `GET /metrics` response

```json
{
  "total_incidents": 12, "critical_incidents": 2, "high_incidents": 7,
  "pending_approvals": 3, "correlated_incidents": 1,
  "by_severity": {"HIGH": 7, "LOW": 3, "CRITICAL": 2},
  "by_attack_type": {"Brute Force": 5, "Phishing": 4},
  "by_approval_status": {"PENDING": 3, "APPROVED": 9},
  "average_latency_seconds": 0.42
}
```

---

## 3. Pages

Five pages, selected from a persistent sidebar.

### 3.1 Dashboard

The landing view. Answers "what is happening right now?"

- **Five metric tiles**: total incidents, critical, high, awaiting approval,
  correlated.
- **Two charts**: attack-type distribution, severity distribution. Bar charts in
  the current build; anything readable is fine.
- **Recent incidents table**: incident id, timestamp, attack type, severity,
  risk score, approval status, truncated input preview. Rows should link to the
  full report.
- **Footer stat**: average end-to-end analysis latency.

Empty state matters — a fresh install has zero incidents. Show a prompt to
analyse something, not a blank grid.

### 3.2 Analyse incident

The primary working view.

**Input area**
- A demo-scenario picker that populates the text box. Six scenarios ship with
  the system: phishing email, brute force, SQL injection, multi-stage intrusion,
  benign activity, insufficient evidence. Keep this — it is how the system gets
  demonstrated.
- A large multi-line text area. Placeholder should state that it accepts an
  alert, a log excerpt, an email, a URL, or several events separated by blank
  lines or `Event N:` headers.
- Three options: retrieval on/off toggle, asset-criticality toggle, and a 1-5
  slider shown only when that toggle is on.
- A primary **Analyse** button, disabled while the box is empty.

**Result area — two mutually exclusive states**

*State A: awaiting approval.* This is the most important screen in the product.
- A prominent warning banner carrying `pending_approval.reason`.
- Four tiles: attack type, risk level (colour-coded), risk score, confidence.
- Left column: evidence list, grounded technique ids.
- Right column: proposed recommendations, with high-impact ones marked.
- A distinct, visually stronger callout listing `high_impact_actions` — these
  are what the analyst is actually deciding about.
- Retrieved sources with links.
- An optional decision note field.
- Three buttons: **Approve**, **Reject**, **Escalate**.
- A caption stating the workflow is paused and nothing executes on any outcome.

*State B: completed report.* Header tiles (attack type, severity, confidence,
risk score), the summary paragraph, then tabbed detail:

| Tab | Contents |
|---|---|
| Evidence | Evidence list plus the model's reasoning |
| Threat intelligence | Grounded MITRE techniques (linked), CWE, CVE, **rejected claims warning**, retrieved sources with expandable text and similarity scores |
| Risk | Likelihood, impact, score tiles; the formula; the full derivation rationale as a list |
| Recommendations | Each action with rationale, priority, and an approval marker; the disclaimer |
| Attack chain | Stage progression, or an explanation of why no chain is shown |
| Explainability | The eight question/answer pairs |
| Raw JSON | The complete report object |

**Below both states**
- *Incident memory*: if `history_matches` is non-empty, show which indicators
  have appeared in previous incidents and how many.
- *Errors*: if `errors` is non-empty, list them. A degraded run still produces a
  report; the user must be able to see what degraded.
- *Execution path*: `node_path` rendered as a breadcrumb. This is how a viewer
  sees which route the workflow took.

### 3.3 Threat intelligence

A search box over the knowledge base, with a results-count slider. Results show
source, document id, title, link, expandable retrieved text and similarity
score. Caption states these are the same documents the analysis pipeline uses
for grounding.

### 3.4 Approvals

A queue of incidents with `approval_status == "PENDING"`. Each card: incident
id, attack type, severity badge, timestamp, risk score, input preview. Empty
state is a success message.

*Known weakness in the current build:* you cannot act on an item directly from
this queue — it tells you to go back to the Analyse page. **A new UI should fix
this** and allow approving from the queue. The API already supports it: call
`POST /approval/{incident_id}` with the incident id.

### 3.5 Methodology

Static explanatory page: the pipeline order, what the model does and does not
decide, the grounding rule, the 5×5 risk matrix as a table, the approval policy,
and the limitations. This page exists so a reviewer can understand the system
without reading code. Keep an equivalent.

---

## 4. Cross-cutting elements

**Sidebar (always visible)**
- Version number.
- API base URL, editable.
- Backend connectivity indicator plus a re-check button.
- Current configuration: LLM backend, embedding backend, adapter path.
- Page navigation.

**Severity colour scale** — used consistently everywhere:

| Level | Colour |
|---|---|
| CRITICAL | `#b3261e` |
| HIGH | `#e8590c` |
| MEDIUM | `#b58900` |
| LOW | `#2f7d32` |
| UNKNOWN | `#5f6368` |

**Loading state.** Analysis takes from ~0.3 s (deterministic backend) to ~30 s
(3B model on a laptop GPU). A spinner with a meaningful label is required, not
optional.

---

## 5. Rules a replacement UI must not break

These are product requirements, not styling preferences. Each exists because of
a specific failure mode.

1. **Never imply the system acted.** Recommendations are advice. No button may
   be labelled in a way that suggests it blocks, isolates or disables anything.
   Approval decides what appears in the report — nothing else.

2. **Always show the disclaimer** on any completed report.

3. **Always surface `mitre.rejected_claims`.** These are identifiers the model
   proposed that retrieval did not support. They are shown, not hidden, because
   visible rejection is the evidence that grounding works. Present them as a
   warning, clearly separated from accepted findings.

4. **Never show a threat-intelligence id without its source and link.** Every
   technique, CWE and CVE arrives with a URL. Render it.

5. **Show risk derivation, not just the number.** `risk.rationale` explains how
   the score was reached. A bare "15" is not defensible; the derivation is.

6. **Distinguish "no threat found" from "analysis failed".** `attack_type:
   "Unknown"` with empty evidence means insufficient evidence — a legitimate,
   deliberate outcome. It must not look like an error.

7. **Present the attack chain as a hypothesis.** Render `correlation.caveat`
   alongside any chain. Never present it as a confirmed intrusion.

8. **Show `errors` when present.** The system degrades rather than failing; the
   user must know when it did.

---

## 6. Honest assessment of the current UI

What works: correct information architecture, everything traceable, safety rules
enforced, all states handled.

What a new UI should improve:

- **Visual design is minimal.** Default Streamlit components throughout. No
  design system, no considered typography, no spacing rhythm.
- **The approvals queue is not actionable.** Biggest functional gap.
- **No real-time feedback during analysis.** The workflow runs seven-plus nodes
  and the user sees only a spinner. Streaming node progress would be a genuine
  improvement.
- **Attack chain is a text arrow sequence.** A proper graph visualisation would
  communicate far better.
- **No filtering or search on the incident list.** The API supports filters that
  the UI does not expose.
- **No dark mode, no responsive layout, no keyboard navigation.**
- **Charts are default bar charts.** No shared colour language between charts
  and severity badges.

---

## 7. Suggested prompt for a UI builder

> Build a SOC analyst dashboard for an AI-assisted security incident analysis
> system. It consumes the REST API described in `docs/ui_spec.md` (OpenAPI at
> `/openapi.json`). Five views: dashboard, analyse incident, threat-intelligence
> search, approvals queue, methodology.
>
> The defining interaction: submitting an event may return a *paused* analysis
> requiring human approval before completion. In that state show the evidence,
> confidence, risk derivation, retrieved sources, and specifically which
> recommended actions would change production state — then offer Approve,
> Reject and Escalate. Nothing is ever executed by the system; approval only
> determines what appears in the final report.
>
> Follow the rules in section 5 of the specification exactly; they are safety
> requirements. Improve on section 6.
