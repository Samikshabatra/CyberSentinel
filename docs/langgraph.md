# LangGraph orchestration

## 1. Why a graph and not a chain

A linear chain would run every node for every input. This workflow needs
decisions that a chain cannot express:

- a benign event must **skip** retrieval entirely - querying threat intelligence
  for a non-finding wastes latency and invites the model to attach a technique
  to nothing;
- a multi-event submission must reach correlation with per-event detections
  intact;
- a high-risk finding must **stop** and wait for a human, then resume;
- a rejected recommendation must **loop back** for re-analysis, exactly once.

Stopping mid-run and resuming later requires persisted state. That is what
LangGraph's checkpointer provides, and it is the reason it is used here rather
than as decoration.

## 2. State

`CyberState` (`graph/state.py`) is a `TypedDict`. Nodes return **partial**
updates that LangGraph merges. Three channels use an append reducer so
concurrent or repeated writes accumulate rather than overwrite:

```python
messages:   Annotated[list[dict], append_list]
errors:     Annotated[list[str],  append_list]
node_trace: Annotated[list[NodeTrace], append_list]
```

`node_trace` is what the agent-workflow evaluation measures and what the UI
renders as the execution path. Everything else is last-write-wins, which is
correct for analysis results that later nodes refine.

## 3. Nodes

| Node | Kind | Responsibility |
|---|---|---|
| `input_classifier` | deterministic | Detect format, split events, extract indicators |
| `threat_detector` | LLM | Classify each event, extract evidence |
| `threat_intelligence` | RAG + LLM | Retrieve and ground identifiers |
| `correlation` | deterministic + LLM | Shared indicators, kill-chain ordering, summary |
| `risk_assessment` | deterministic | Likelihood × impact |
| `response_recommendation` | LLM + filter | Defensive actions, safety-filtered |
| `approval_gate` | deterministic | Decide whether sign-off is required |
| `human_approval` | interrupt | Apply the analyst decision |
| `escalation` | deterministic | Record rejection/escalation, bound the loop |
| `incident_report` | LLM + assembly | Final structured report |

Input classification is deterministic on purpose. Routing is control flow: it
should be reproducible and free. An LLM call there would add latency and
non-determinism to every run without improving the decision.

### No node raises

Every node is wrapped by `node_guard`, which converts an unhandled exception
into an entry in `errors` plus a trace record with `status: "error"`. A
retrieval outage degrades the report; it does not lose the analysis. This is
covered by `test_node_failure_is_captured_not_raised`.

## 4. Conditional edges

All routers are pure functions of state in `graph/edges.py`, so they are
directly unit-testable without running the graph.

```
START ──► input_classifier
             │ route_after_classification: alert | email | url | log |
             │                             vulnerability | multi_event
             ▼
        threat_detector
             │ route_after_detection
             ├── skip_intel  (benign or unknown) ──┐
             └── intel ──► threat_intelligence ────┤
                                                   ▼
                                              correlation
                                                   ▼
                                            risk_assessment
                                                   │ route_after_risk
                                                   ▼
                                        response_recommendation
                                                   ▼
                                             approval_gate
                                                   │ route_after_gate
                          ┌────────────────────────┴──────────┐
                    required                            not required
                          ▼                                   │
              [INTERRUPT] human_approval                      │
                          │ route_after_approval              │
          ┌───────────────┼──────────────┐                    │
      APPROVE          REJECT        ESCALATE                 │
          │               ▼              ▼                    │
          │          escalation ──► escalation                │
          │               │              │                    │
          │      route_after_escalation  │                    │
          │               ▼              │                    │
          │   response_recommendation ───┘                    │
          ▼                                                   ▼
                        incident_report ──► END
```

## 5. Human-in-the-loop

The graph is compiled with a checkpointer and
`interrupt_before=["human_approval"]`. Execution genuinely stops before that
node; `graph.get_state(config).next` contains `human_approval` while paused.

```python
run = workflow.analyze(event)          # pauses if approval is required
run.awaiting_approval                  # True
run.state["final_report"]              # absent - no report is produced yet

resumed = workflow.submit_decision(run.thread_id, "APPROVED", decided_by="analyst")
resumed.report["approval"]["decision"] # "APPROVED"
```

Over HTTP the same flow is `POST /analyze` followed by
`POST /approval/{incident_id}`, keyed by thread id. The thread id is the
incident id, so a paused run is resumable by a later request - including from a
different client.

### When approval is required

1. Assessed risk reaches the configured threshold (`HIGH` by default), or
2. any recommendation would change production state - block, isolate, disable,
   revoke, quarantine, terminate - at **any** risk level.

The second rule matters: a MEDIUM-risk finding whose recommended action is
"block the source IP" still needs a human.

### The three decisions

| Decision | Effect |
|---|---|
| `APPROVED` | Report is finalised with all recommendations intact |
| `REJECTED` | High-impact actions are withdrawn, incident is re-analysed once for investigative steps only |
| `ESCALATED` | Recorded for senior review, no action taken |

Nothing is executed in any case. Approval determines which recommendations
appear in the report, not what happens to the environment.

### Bounding the loop

Rejection routes back through `escalation` to `response_recommendation`, which
would return to `approval_gate` and interrupt again - forever. Two guards
prevent this:

- `approval_gate` treats a decision that has already been recorded as final for
  the run and sets `required = False`;
- `MAX_REANALYSIS` caps re-analysis passes at one.

Both are covered by tests (`test_rejection_loop_is_bounded`,
`test_rejection_withdraws_disruptive_actions`).

## 6. Observability

Every node logs `run_id`, `incident_id`, `node`, `status` and `latency_s`:

```
01:37:56 INFO cybersentinel.graph.nodes | node completed
    [run_id=ccadd02f5123 incident_id=INC-20260808-D4768A node=threat_detector
     status=success latency_s=0.004]
```

The same information is in `state["node_trace"]`, which the API returns as
`node_path` and the evaluation uses to measure routing accuracy. Analyst input
is never logged in full - only a redacted, truncated preview.

## 7. Testing

Routers are pure functions and are tested directly. The workflow is tested end
to end with the mock backend and a local vector store, covering: routing per
input type, the approval pause, all three decisions, the bounded rejection loop,
node failure isolation, and multi-event correlation. See `tests/test_graph.py`.

## 8. Measured behaviour

From `scripts/evaluate.py`:

| Metric | Value |
|---|---|
| Routing accuracy | 1.00 |
| Approval-gate accuracy | 1.00 |
| Workflow completion rate | 1.00 |
| Structured output rate | 1.00 |

Measured over the routing cases in `evaluation/runner.py`, which cover every
input type and both approval outcomes.
