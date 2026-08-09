# Security

## 1. Posture

CyberSentinel is a **defensive analysis system**. It reads descriptions of
security events and produces recommendations for a human analyst. It is not an
offensive tool, not a scanner, and not a response automation platform.

Three rules define the boundary:

1. **Nothing is executed.** No recommendation is ever carried out by the system.
2. **Nothing is fetched.** URLs, domains and hashes in analyst input are parsed
   as text and never contacted.
3. **Nothing is asserted without support.** Threat-intelligence identifiers come
   from retrieval; unsupported claims are rejected and reported.

Each is enforced in code and covered by tests in `tests/test_security.py`.

## 2. Input handling

Analyst input is untrusted data. It is never treated as code.

| Control | Implementation |
|---|---|
| No dynamic execution | No `eval`, `exec`, `subprocess`, `os.system` or `shell=True` anywhere in the package - asserted by a test that scans the source |
| No deserialisation of untrusted data | No `pickle.load` - asserted by a test |
| Control-character stripping | `sanitize_text` removes control characters and normalises newlines |
| Length limits | `MAX_INPUT_CHARS` (20,000) on text, `MAX_UPLOAD_BYTES` (1 MB) on uploads |
| Empty input rejection | Rejected at the workflow boundary with a typed error, not deep inside a node |
| Schema validation | Pydantic models with `extra="forbid"`, bounded confidence, closed severity set |

Payloads such as `'; DROP TABLE incidents; --`, `<script>alert(1)</script>`,
`$(curl … | sh)` and `__import__('os').system('whoami')` are submitted in the
test suite and asserted to be *classified*, never interpreted.

## 3. No outbound requests to analyst-supplied destinations

A URL submitted for analysis is examined textually - structure, lookalike
domains, defanging, registration patterns described in the input. It is never
requested. `test_url_analysis_does_not_fetch_the_url` monkeypatches `httpx` to
raise on any outbound call and runs a URL through the full workflow.

Defanged indicators (`hxxp://bad[.]example[.]net`) are refanged only to make
pattern matching work; refanging is string manipulation.

The only hardcoded external URL in the system is MITRE's own `mitre/cti`
repository, contacted only when `--fetch-attack` is passed explicitly to the
ingestion script.

## 4. Recommendation safety

Recommendations pass two filters before an analyst sees them:

**Prohibited actions are dropped.** Offensive or evidence-destroying actions are
matched by pattern and removed even if the model proposes them: hacking back,
counter-attacking, running a scan or penetration test against a source, deleting
or clearing logs, disabling logging or auditing, paying a ransom, wiping or
formatting a system. `test_prohibited_model_output_is_filtered_out` feeds the
pipeline a model response containing "Hack back the attacker" and asserts it
does not reach the output.

**High-impact actions are flagged.** Anything that would block, isolate,
quarantine, disable, suspend, revoke, terminate or reimage is marked
`high_impact` and forces `requires_approval`, regardless of risk level.

A sweep test runs every attack category through the recommender and asserts no
generated recommendation is ever prohibited.

## 5. Human-in-the-loop

Analyst approval is required when:

- assessed risk reaches `APPROVAL_SEVERITY_THRESHOLD` (default `HIGH`), **or**
- any recommendation would change production state, at any risk level.

The workflow genuinely pauses at a LangGraph checkpoint. While paused, no report
exists and no action is proposed as taken. The analyst sees the evidence, the
confidence, the risk derivation, the grounded sources and the specific
high-impact actions before deciding.

`APPROVED`, `REJECTED` and `ESCALATED` all leave the environment untouched. The
decision determines which recommendations appear in the report.

## 6. Secrets and data handling

| Control | Implementation |
|---|---|
| No hardcoded secrets | Enforced by a source-scanning test |
| `.env` is git-ignored | Asserted by a test |
| `.env.example` contains no values for credential keys | Asserted by a test |
| Log redaction | Passwords, API keys, bearer tokens and email addresses are redacted before any log line is written |
| Truncated storage | Only a 500-character redacted preview of the input is persisted, never the full submission |
| Minimal indicator storage | Extracted indicators are stored for history lookups; raw log bodies are not |

`test_repository_input_preview_is_redacted` submits
`password=hunter2 ... user@example.com`, persists it, and asserts neither value
appears in the stored preview.

## 7. Service hardening

- The container runs as an unprivileged user (uid 10001).
- Container images carry health checks; `GET /health` reports each component and
  whether it is running on a fallback.
- Unhandled API exceptions return a generic message; the full trace goes to the
  logs only.
- Request validation errors are serialised without leaking internal exception
  objects.
- CORS is permissive only when `APP_ENV=development`.

## 8. Known limitations

- **No authentication.** The API has no authN/authZ. It is a project system, not
  a deployed service. Anything internet-facing needs an authentication layer and
  rate limiting in front of it.
- **PostgreSQL default credentials** in `docker-compose.yml` are development
  defaults and must be overridden for any shared deployment.
- **No multi-tenancy.** Incident history is a single shared namespace.
- **No audit log of reads.** Approval decisions are audited; queries are not.
- **Prompt injection is not fully solved.** Analyst input reaches the model, so
  crafted text could attempt to influence it. Structural controls limit the
  blast radius: identifiers must come back from retrieval, risk is computed in
  Python, prohibited actions are filtered, and high-impact actions require human
  approval. A successful injection can distort a classification; it cannot make
  the system execute anything.

## 9. Reporting language

The system never claims certainty. Reports carry a standing disclaimer and use
hedged language ("probable", "consistent with", "requires analyst validation").

A test (`test_reports_do_not_overclaim`) scans generated reports for absolute
claims - perfect accuracy, full autonomy, guaranteed results, absence of
hallucination - and fails if any appears. A second test
(`test_docs_avoid_absolute_claims`) applies the same rule to the README and
every file in `docs/`, this one included.

## 10. Responsible use

This project is for education, defensive analysis and research. Use it only on
data you are authorised to handle. It does not replace a security team, and its
output is a starting point for investigation, not a conclusion.
