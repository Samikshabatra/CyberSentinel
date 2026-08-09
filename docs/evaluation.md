# Evaluation

## 1. Running it

```bash
python scripts/prepare_dataset.py                  # build the test split
python scripts/ingest_knowledge_base.py --local    # build the knowledge base
python scripts/evaluate.py                         # rules baseline + all studies
python scripts/evaluate.py --limit 40              # quick pass
python scripts/evaluate.py --with-hf --adapter models/cybersentinel-lora
```

Results are written to `evaluation/results/`: a timestamped JSON file, a
`latest.json`, and a Markdown summary. Everything needed to reproduce a figure is
in the JSON, including the environment (backend, adapter, embedding backend,
vector store) the run used.

## 2. What is measured

| Study | Question |
|---|---|
| Experiment 1 | Does the model classify correctly on held-out data? |
| Experiment 1b | Does it still work when the surface vocabulary is removed? |
| Experiment 2 | Does RAG improve grounding and reduce unsupported claims? |
| Experiment 3 | What does each pipeline layer contribute? |
| Hallucination | Does the system decline when evidence is insufficient? |
| Retrieval | Does the retriever find the right documents? |
| Agent workflow | Does the orchestrator route and gate correctly? |

## 3. Metrics and why these ones

**Macro F1 is the headline.** The test set is balanced by construction, but
macro F1 keeps a rare category from being hidden by a common one - the failure
mode that matters in a SOC. Macro averages are computed over the classes
actually present in a run; averaging in classes that never appear would drag the
score down for a reason unrelated to model quality.

**Structural metrics are first-class.** A model that classifies well but emits
unparseable JSON is useless in a pipeline, so JSON validity, field completeness
and severity accuracy are reported alongside classification, not as footnotes.

**Severity within one band** is reported alongside exact severity accuracy. A
HIGH/CRITICAL mix-up is operationally very different from a LOW/CRITICAL one,
and exact match alone hides that distinction.

## 4. Experiment 1 - detection quality

Arms are evaluated on the identical held-out test split with identical prompts,
parsing and scoring; only the model serving detection differs.

| Arm | Accuracy | Macro F1 | JSON valid | Severity acc. |
|---|---|---|---|---|
| rules_baseline | 1.00 | 1.00 | 1.00 | 0.67 |

**This is a ceiling result and it must be reported as one.** The generated test
set is drawn from the same template grammar as the training data, so a keyword
system solves it. No arm can demonstrate an advantage here. Severity accuracy of
0.67 is the one signal that survives: severity is genuinely harder than category,
because the same attack class carries different severities in different contexts.

## 5. Experiment 1b - hard test set

Hand-authored cases: paraphrased attacks with no signature vocabulary, benign
events full of alarming keywords (an authorised penetration test, a scheduled
vulnerability scan, a `DROP TABLE` inside an approved change window), and inputs
where the correct answer is `Unknown`.

| Arm | Accuracy | Macro F1 |
|---|---|---|
| rules_baseline | 0.11 | 0.01 |

The drop from 1.00 to 0.11 is the honest measure of how much of the generated
benchmark is surface pattern matching. This is the set on which a fine-tuned
model has room to show benefit, and the set whose numbers should be quoted when
comparing arms.

## 6. Experiment 2 - retrieval grounding

The same pipeline with retrieval disabled and enabled.

| Arm | Grounded | Citations | MITRE recall | Rejected claims / example |
|---|---|---|---|---|
| without_rag | 0.00 | 0.00 | 0.00 | 0.00 |
| with_rag | 1.00 | 1.00 | 1.00 | 0.00 |

Without retrieval the system reports no threat-intelligence identifiers at all -
which is the designed behaviour, not a failure. The model is never permitted to
supply an identifier from memory, so with RAG disabled there is nothing to
ground and nothing is claimed. This is the concrete demonstration that grounding
is enforced structurally rather than requested in a prompt.

## 7. Experiment 3 - pipeline ablation

| Arm | Accuracy | Grounded | Recommendations | Failure rate | Mean latency (s) |
|---|---|---|---|---|---|
| 1_single_call_rules | 1.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| 3_detection_plus_rag | 1.00 | 1.00 | 0.00 | 0.00 | 0.004 |
| 4_full_langgraph_workflow | 1.00 | 1.00 | 1.00 | 0.00 | 0.011 |

Classification accuracy is identical across arms, as expected - the same
detector runs in all of them. What each layer adds is visible in the other
columns: retrieval adds grounding and citations, and the full workflow adds risk
scoring, correlation, approval gating and actionable recommendations. The cost is
about 11 ms per analysis with the mock backend, which is where the orchestration
overhead can be read directly, uncontaminated by model latency.

With `--with-hf`, arms `0_base_model_single_call` and `2_finetuned_single_call`
are added and the classification column becomes informative.

## 8. Hallucination study

Two adversarial input sets: twelve inputs with no actionable evidence, and four
leading questions that explicitly demand a CVE or technique the evidence cannot
support ("Which CVE explains slow response times on the website?").

| Metric | Value | Direction |
|---|---|---|
| Insufficient-evidence rate (adversarial) | 1.00 | higher is better |
| Claims asserted (adversarial) | 0.00 | lower is better |
| Claims asserted (leading questions) | 0.00 | lower is better |
| CVE claims blocked by the grounding filter | 0 | — |

The system returned `Unknown` for every evidence-free input and asserted no
identifier under direct pressure to produce one. The blocked-CVE count is zero
because the mock backend never proposes an ungrounded identifier in the first
place; the filter's rejection path is exercised directly in
`tests/test_rag.py::test_cve_must_appear_in_context` and
`test_invented_technique_is_rejected`.

## 9. Retrieval study

Relevance ground truth is the verified category-to-technique/CWE mapping, so the
study is reproducible without human annotation.

| Metric | Value |
|---|---|
| Mean precision@5 | 0.77 |
| Mean recall@5 | 0.96 |
| Mean reciprocal rank | 1.00 |

MRR of 1.00 means the correct document ranked first for every category.
Precision is below 1.0 because a top-5 query legitimately returns related
documents outside the narrow ground-truth set - a brute-force query also
surfaces CWE-307 and authentication guidance, which is useful, not wrong.

## 10. Agent workflow study

| Metric | Value |
|---|---|
| Routing accuracy | 1.00 |
| Approval-gate accuracy | 1.00 |
| Workflow completion rate | 1.00 |
| Structured output rate | 1.00 |

Covers all six input types and both approval outcomes. Routing accuracy started
at 0.86: emails were being split into two "events" at the blank line between
headers and body and routed as multi-event. The classifier now recognises
single-document formats before testing for multiple events, and the case is
locked down by `test_email_is_not_split_into_multiple_events`.

## 11. Performance

Latency is recorded per node and aggregated per run. With the mock backend the
full pipeline is about 11 ms, dominated by retrieval. With a real model,
detection dominates and the orchestration overhead stays roughly constant, so
the ablation table's latency column is the honest cost of the graph itself.

Token usage is estimated from character counts (≈4 characters per token) and
reported per arm as `approx_tokens_per_example`. It is an estimate, labelled as
one, and is used for relative comparison between arms rather than for billing.

## 12. Research questions

| RQ | Where it is answered | Current status |
|---|---|---|
| RQ1 - does fine-tuning improve classification? | Experiments 1 and 1b | Harness complete; run `--with-hf` with a trained adapter. The generated set is saturated; report the hard set. |
| RQ2 - does RAG improve grounding? | Experiment 2 | Answered: grounded rate 0.00 → 1.00, MITRE recall 0.00 → 1.00 |
| RQ3 - does orchestration improve multi-step analysis? | Experiment 3 | Answered: correlation, risk, approval and recommendations appear only in the full arm, at ~11 ms cost |
| RQ4 - can correlation identify multi-stage attacks? | Agent study, `test_multi_event_correlation_and_chain` | Answered: four related events resolve to Reconnaissance → Credential Access → Privilege Escalation at CRITICAL risk |
| RQ5 - does human-in-the-loop improve safety? | Agent study, approval tests | Answered: gate accuracy 1.00; no disruptive action is ever taken without an explicit decision |

## 13. Threats to validity

- **The generated test set is saturated.** Template-derived text is separable by
  keywords. Conclusions about model quality should be drawn from the hard set.
- **The hard set is small** (36 cases) and authored by the same person who built
  the system. It is a probe, not a benchmark.
- **The default arm is a rules baseline**, not an LLM. Numbers labelled
  `rules_baseline` describe a keyword system.
- **Retrieval ground truth is taxonomy-derived**, so it measures whether
  retrieval agrees with the project's own mapping, not with an external
  authority.
- **Token counts are estimates**, not tokenizer output.
- **No statistical significance testing.** Single runs on a small test set; the
  differences reported are descriptive.

## 14. Reproducing a run

```bash
python scripts/prepare_dataset.py --seed 20260808
python scripts/ingest_knowledge_base.py --local --reset
python scripts/evaluate.py --save-predictions
```

Dataset generation, template splitting and the mock backend are all
deterministic, so these three commands reproduce the reported numbers exactly.
`--save-predictions` includes every per-example prediction in the JSON for error
analysis.
