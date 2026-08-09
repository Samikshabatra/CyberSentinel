# Screenshots

Captured from the running console. Each is listed with the point it demonstrates,
so they can be dropped straight into a report with a caption.

All eight were taken with `LLM_BACKEND=mock`, the deterministic rules analyst.
Classifications therefore come from keyword scoring, not the fine-tuned model —
the interface, workflow and grounding behaviour are identical either way. Any
caption should say so rather than implying the model produced them.

| # | File | Demonstrates |
|---|---|---|
| 1 | `01-overview-dashboard.jpg` | Incident volume, severity mix, attack-type distribution and recent incidents |
| 2 | `02-approval-checkpoint-brute-force.jpg` | The human-in-the-loop checkpoint: evidence, grounded techniques, and the single action requiring a decision |
| 3 | `03-incident-report-approved.jpg` | Completed report after approval, with incident memory linking the source IP to prior incidents |
| 4 | `04-grounded-threat-intelligence.jpg` | T1110 and sub-techniques with tactic and source links, plus the CWE mapping |
| 5 | `05-multistage-critical-checkpoint.jpg` | Four correlated events resolving to Privilege Escalation at CRITICAL, risk 25/25 |
| 6 | `06-attack-chain.jpg` | Reconnaissance → Credential Access → Privilege Escalation, presented as a hypothesis |
| 7 | `07-insufficient-evidence-unknown.jpg` | A vague input returning `Unknown` at confidence 0.00 instead of a guess |
| 8 | `08-methodology-risk-matrix.jpg` | The 5×5 risk matrix, approval policy and stated limitations |

## The two that carry a viva

**7 — insufficient evidence.** A vague report produces `Unknown`, confidence
0.00, risk 2, and an explicit statement that no classification is claimed. The
node path also shows `threat_intelligence` skipped: with nothing to ground,
retrieval is not run and no identifier is asserted. This is the system declining
to invent a finding, which is the behaviour most demos cannot show.

**8 — methodology.** Risk is arithmetic on a documented matrix, not a model
opinion, and the page states the limitations rather than burying them.

## Reproducing them

```bash
streamlit run app/streamlit_app.py
```

Each image maps to a demo scenario in the Analyse incident picker: 2 and 3 use
"Brute force", 5 and 6 use "Multi-stage intrusion", 7 uses "Insufficient
evidence".
