"""CyberSentinel SOC console.

A dark operations-centre interface over the analysis API. The visual system is
in `theme.py`; this module is structure and behaviour.

The interface talks to the FastAPI backend when one is reachable and falls back
to calling the service layer in-process when it is not, so a single command is
enough during development.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme  # noqa: E402

from cybersentinel.cybersecurity.risk import (  # noqa: E402
    IMPACT_SCALE,
    LIKELIHOOD_SCALE,
    RISK_BANDS,
)
from cybersentinel.utils.config import get_settings  # noqa: E402

#: Navigation order. No icons: the reference console keeps its nav typographic,
#: and emoji would be the loudest thing on an otherwise restrained panel.
PAGES = [
    "Overview",
    "Analyse incident",
    "Threat intelligence",
    "Approvals",
    "Methodology",
]

DEMO_SCENARIOS: dict[str, str] = {
    "Phishing email": (
        "From: security-alert@example.com\n"
        "Subject: Urgent - verify your account within 24 hours\n\n"
        "Your mailbox will be suspended unless you confirm your credentials at "
        "http://secure-example.com.verify-now.example.net/login"
    ),
    "Brute force": (
        "47 failed SSH login attempts from 198.51.100.23 within 3 minutes, all targeting "
        "the account root."
    ),
    "SQL injection": (
        "Web server log: GET /products?id=1' OR '1'='1 UNION SELECT null,username,password "
        "FROM users-- HTTP/1.1 from 203.0.113.45 returned 200 with an unusually large "
        "response body."
    ),
    "Multi-stage intrusion": (
        "Event 1: Port scan from 203.0.113.45 against 1200 sequential ports on web-prod-01\n\n"
        "Event 2: 20 failed SSH logins for user admin from 203.0.113.45\n\n"
        "Event 3: Successful SSH login for user admin from 203.0.113.45\n\n"
        "Event 4: User admin added to the administrators group shortly after login"
    ),
    "Benign activity": (
        "User alice authenticated successfully to the VPN at 08:52 from the usual office "
        "address, matching their normal weekday pattern. Routine activity."
    ),
    "Insufficient evidence": "Something looks wrong with the server, please investigate.",
}


# --------------------------------------------------------------------------- #
# Backend access
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_local_service() -> Any:
    from cybersentinel.service import AnalysisService

    return AnalysisService()


def api_base() -> str:
    return st.session_state.get("api_base", get_settings().api_base_url).rstrip("/")


def api_available() -> bool:
    import httpx

    try:
        return httpx.get(f"{api_base()}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


def call_api(method: str, path: str, **kwargs: Any) -> Any:
    import httpx

    response = httpx.request(method, f"{api_base()}{path}", timeout=180.0, **kwargs)
    response.raise_for_status()
    return response.json()


def analyse(text: str, use_rag: bool, asset_criticality: int | None) -> dict[str, Any]:
    payload = {
        "text": text,
        "use_rag": use_rag,
        "use_llm_response": True,
        "asset_criticality": asset_criticality,
    }
    if st.session_state.get("use_api"):
        return call_api("POST", "/analyze", json=payload)

    result = get_local_service().analyze(
        text, use_rag=use_rag, use_llm_response=True, asset_criticality=asset_criticality
    )
    payload = result.to_dict()
    if result.awaiting_approval:
        state = result.state
        approval = state.get("approval") or {}
        analysis = state.get("threat_analysis") or {}
        risk = state.get("risk_assessment") or {}
        recommendations = state.get("response_recommendations") or []
        payload["pending_approval"] = {
            "reason": approval.get("reason", ""),
            "attack_type": analysis.get("attack_type"),
            "severity": analysis.get("severity"),
            "confidence": analysis.get("confidence", 0.0),
            "risk_level": risk.get("risk_level"),
            "risk_score": risk.get("risk_score"),
            "evidence": analysis.get("evidence", []),
            "recommendations": recommendations,
            "high_impact_actions": [
                item["action"] for item in recommendations if item.get("high_impact")
            ],
            "sources": (state.get("retrieved_context") or [])[:5],
            "mitre_techniques": [
                technique["technique_id"]
                for technique in (state.get("mitre_mapping") or {}).get("techniques", [])
            ],
        }
    return payload


def submit_decision(thread_id: str, decision: str, note: str | None) -> dict[str, Any]:
    if st.session_state.get("use_api"):
        return call_api(
            "POST",
            f"/approval/{thread_id}",
            json={"decision": decision, "decided_by": "console-analyst", "note": note},
        )
    return get_local_service().submit_decision(
        thread_id, decision, decided_by="console-analyst", note=note
    ).to_dict()


def fetch_metrics() -> dict[str, Any]:
    if st.session_state.get("use_api"):
        return call_api("GET", "/metrics")
    return get_local_service().metrics()


def fetch_incidents(limit: int = 50) -> list[dict[str, Any]]:
    if st.session_state.get("use_api"):
        return call_api("GET", "/incidents", params={"limit": limit})
    return get_local_service().list_incidents(limit=limit)


def fetch_pending() -> list[dict[str, Any]]:
    if st.session_state.get("use_api"):
        return call_api("GET", "/incidents/pending-approval")
    return get_local_service().pending_approvals()


def search_intel(query: str, top_k: int) -> dict[str, Any]:
    if st.session_state.get("use_api"):
        return call_api("GET", "/threat-intelligence/search", params={"q": query, "top_k": top_k})

    from cybersentinel.rag.retriever import Retriever

    retriever = st.session_state.setdefault("retriever", Retriever())
    result = retriever.retrieve(query, top_k=top_k)
    return {
        "query": result.query,
        "store": result.store,
        "latency_seconds": result.latency_seconds,
        "documents": [document.model_dump(mode="json") for document in result.documents],
        "error": result.error,
    }


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict[str, Any]], empty_note: str = "") -> None:
    if not sources:
        st.markdown(
            f'<div class="cs-evidence">{empty_note or "No sources were retrieved, so no identifier is asserted."}</div>',
            unsafe_allow_html=True,
        )
        return

    for source in sources:
        identifier = source.get("document_id") or ""
        title = source.get("title") or ""
        url = source.get("url")
        score = source.get("score")
        meta = f"{source.get('source', '')}"
        if score is not None:
            meta += f" · similarity {score}"

        link = f'<a href="{url}" target="_blank" style="color:inherit">{title}</a>' if url else title
        st.markdown(
            f'<div class="cs-source"><div class="cs-source-head">'
            f'<span class="cs-source-id">{identifier}</span>'
            f'<span class="cs-source-meta">{meta}</span></div>'
            f'<div class="cs-source-title">{link}</div></div>',
            unsafe_allow_html=True,
        )
        content = source.get("content")
        if content:
            with st.expander("Retrieved text"):
                st.write(content)


def render_report(report: dict[str, Any]) -> None:
    """A completed incident report."""
    risk = report.get("risk") or {}
    severity = report.get("severity", "UNKNOWN")

    columns = st.columns(4)
    columns[0].markdown(
        theme.tile("Attack type", report.get("attack_type", "Unknown"),
                   colour=theme.severity_colour(severity)),
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        theme.tile("Severity", severity, colour=theme.severity_colour(severity)),
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        theme.tile("Confidence", f"{float(report.get('confidence', 0.0)):.2f}",
                   note="model-reported", colour=theme.PALETTE["info"]),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        theme.tile("Risk score", risk.get("risk_score", "—"),
                   note=f"likelihood {risk.get('likelihood', '—')} × impact {risk.get('impact', '—')}",
                   colour=theme.severity_colour(risk.get("risk_level"))),
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(f'<div class="cs-card">{report.get("summary", "")}</div>', unsafe_allow_html=True)
    st.markdown("")

    tabs = st.tabs(
        ["Evidence", "Threat intelligence", "Risk", "Recommendations",
         "Attack chain", "Explainability", "Raw"]
    )

    with tabs[0]:
        st.markdown(theme.evidence_list(report.get("evidence", [])), unsafe_allow_html=True)
        if report.get("reasoning"):
            st.markdown("**Model reasoning**")
            st.write(report["reasoning"])

    with tabs[1]:
        mitre = report.get("mitre") or {}
        techniques = mitre.get("techniques") or []

        if techniques:
            st.markdown("**Grounded in retrieved sources**")
            rows = [
                [t["technique_id"], t["name"], t["tactic"],
                 f'<a href="{t["url"]}" target="_blank" style="color:{theme.PALETTE["info"]}">source</a>']
                for t in techniques
            ]
            st.markdown(
                theme.table(["Technique", "Name", "Tactic", "Reference"], rows),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="cs-evidence">No MITRE technique was supported by the retrieved sources.</div>',
                unsafe_allow_html=True,
            )

        for label, key, id_field in (("Weaknesses", "cwe", "cwe_id"), ("Vulnerabilities", "cve", "cve_id")):
            entries = mitre.get(key) or []
            if entries:
                st.markdown(f"**{label}**")
                st.markdown(
                    "".join(
                        f'<a href="{entry["url"]}" target="_blank" style="text-decoration:none">'
                        f'{theme.tag(entry[id_field])}</a>'
                        for entry in entries
                    ),
                    unsafe_allow_html=True,
                )

        rejected = mitre.get("rejected_claims") or []
        if rejected:
            st.markdown(
                f'<div class="cs-impact"><div class="cs-impact-label">Rejected claims</div>'
                f'<div class="cs-impact-item">The model proposed {", ".join(rejected)}. '
                f"Retrieval did not support these, so they are excluded from the findings.</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Retrieved sources**")
        render_sources(report.get("sources", []))

    with tabs[2]:
        if not risk:
            st.info("No risk assessment was produced.")
        else:
            left, right = st.columns([1, 2])
            with left:
                st.plotly_chart(
                    theme.risk_gauge(int(risk["risk_score"]), risk.get("risk_level", "UNKNOWN")),
                    width='stretch',
                    config={"displayModeBar": False},
                )
            with right:
                st.markdown(
                    theme.table(
                        ["Factor", "Value", "Meaning"],
                        [
                            ["Likelihood", str(risk["likelihood"]), risk.get("likelihood_label", "")],
                            ["Impact", str(risk["impact"]), risk.get("impact_label", "")],
                            ["Score", str(risk["risk_score"]), risk.get("risk_level", "")],
                        ],
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="cs-mono">{risk.get("formula", "")}</div>',
                            unsafe_allow_html=True)

            st.markdown("**How this score was derived**")
            st.markdown(theme.evidence_list(risk.get("rationale", [])), unsafe_allow_html=True)

    with tabs[3]:
        recommendations = report.get("recommendations") or []
        if not recommendations:
            st.info("No recommendations were produced.")
        for item in recommendations:
            marker = (
                f'{theme.badge(item.get("priority", "MEDIUM"))} '
                f'<span class="cs-badge" style="background:{theme.PALETTE["critical"]}1F;'
                f'color:{theme.PALETTE["critical"]};border:1px solid {theme.PALETTE["critical"]}44">'
                f"NEEDS APPROVAL</span>"
                if item.get("high_impact")
                else theme.badge(item.get("priority", "MEDIUM"))
            )
            st.markdown(
                f'<div class="cs-source"><div class="cs-source-head">'
                f'<span class="cs-source-title">{item["action"]}</span>{marker}</div>'
                f'<div class="cs-source-meta" style="margin-top:.3rem">{item.get("rationale", "")}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(report.get("disclaimer", ""))

    with tabs[4]:
        correlation = report.get("correlation") or {}
        chain = correlation.get("attack_chain") or []
        if not chain:
            st.markdown(
                '<div class="cs-evidence">No attack chain is shown. A chain is drawn only when '
                "events share an indicator and span at least two kill-chain stages.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                " ".join(
                    f'{theme.tag(stage["stage"])}<span style="color:{theme.PALETTE["faint"]}">→</span>'
                    if index < len(chain) - 1
                    else theme.tag(stage["stage"])
                    for index, stage in enumerate(chain)
                ),
                unsafe_allow_html=True,
            )
            st.caption(correlation.get("caveat", ""))
            for stage in chain:
                with st.expander(f"{stage['stage']} · events {stage.get('event_indices')}"):
                    st.write(stage.get("description", ""))
                    st.markdown(
                        theme.evidence_list(stage.get("supporting_evidence", [])),
                        unsafe_allow_html=True,
                    )

    with tabs[5]:
        rows = [
            [key.replace("_", " ").title(), value]
            for key, value in (report.get("explainability") or {}).items()
        ]
        st.markdown(theme.table(["Question", "Answer"], rows, id_column=-1), unsafe_allow_html=True)

    with tabs[6]:
        st.json(report)


def render_checkpoint(result: dict[str, Any]) -> None:
    """The approval checkpoint. The one place this interface raises its voice."""
    pending = result.get("pending_approval") or {}
    risk_colour = theme.severity_colour(pending.get("risk_level"))

    st.markdown(
        f'<div class="cs-hold">'
        f'<div class="cs-hold-eyebrow"><span class="cs-dot cs-dot-live" '
        f'style="background:{theme.PALETTE["high"]}"></span>Analysis paused</div>'
        f'<div class="cs-hold-reason">{pending.get("reason", "Analyst approval required.")}</div>'
        f'<div class="cs-hold-note">The workflow is holding at a checkpoint. '
        f"Nothing is executed on any outcome — your decision determines which "
        f"recommendations reach the final report.</div></div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    columns[0].markdown(
        theme.tile("Attack type", pending.get("attack_type", "Unknown"), colour=risk_colour),
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        theme.tile("Risk level", pending.get("risk_level", "—"), colour=risk_colour),
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        theme.tile("Risk score", pending.get("risk_score", "—"), note="of 25", colour=risk_colour),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        theme.tile("Confidence", f"{float(pending.get('confidence', 0.0)):.2f}",
                   colour=theme.PALETTE["info"]),
        unsafe_allow_html=True,
    )

    st.markdown("")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Evidence**")
            st.markdown(theme.evidence_list(pending.get("evidence", [])), unsafe_allow_html=True)
            techniques = pending.get("mitre_techniques") or []
            st.markdown("**Grounded techniques**")
            st.markdown(
                "".join(theme.tag(technique) for technique in techniques)
                or '<div class="cs-mono">none grounded</div>',
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown("**Proposed actions**")
            for item in pending.get("recommendations", []):
                flag = " 🔒" if item.get("high_impact") else ""
                st.markdown(
                    f'<div class="cs-evidence">{item["action"]}{flag}</div>', unsafe_allow_html=True
                )

    high_impact = pending.get("high_impact_actions") or []
    if high_impact:
        st.markdown(
            '<div class="cs-impact"><div class="cs-impact-label">Requires your decision</div>'
            + "".join(f'<div class="cs-impact-item">{action}</div>' for action in high_impact)
            + "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Retrieved sources"):
        render_sources(pending.get("sources", []))

    note = st.text_input("Decision note (optional)", key=f"note_{result['thread_id']}")
    buttons = st.columns([1, 1, 1, 5])
    decision = None
    if buttons[0].button("Approve", type="primary", key=f"approve_{result['thread_id']}"):
        decision = "APPROVED"
    if buttons[1].button("Reject", key=f"reject_{result['thread_id']}"):
        decision = "REJECTED"
    if buttons[2].button("Escalate", key=f"escalate_{result['thread_id']}"):
        decision = "ESCALATED"

    if decision:
        with st.spinner(f"recording {decision.lower()}…"):
            st.session_state["last_result"] = submit_decision(
                result["thread_id"], decision, note or None
            )
        st.rerun()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    theme.page_header(
        "Security overview",
        "Incident volume, severity mix and analyst queue",
        status="Console active",
        status_colour=theme.PALETTE["low"],
    )

    try:
        metrics = fetch_metrics()
    except Exception as exc:
        st.error(f"Metrics unavailable: {type(exc).__name__}: {exc}")
        return

    tiles = [
        ("Incidents analysed", metrics.get("total_incidents", 0), "all time", theme.PALETTE["info"], "📁"),
        ("Critical", metrics.get("critical_incidents", 0), "highest band", theme.PALETTE["critical"], "◆"),
        ("High", metrics.get("high_incidents", 0), "elevated risk", theme.PALETTE["high"], "▲"),
        ("Awaiting decision", metrics.get("pending_approvals", 0), "analyst action", theme.PALETTE["medium"], "🔒"),
        ("Correlated", metrics.get("correlated_incidents", 0), "multi-stage", theme.PALETTE["low"], "⛓"),
    ]
    for column, (label, value, note, colour, icon) in zip(st.columns(5), tiles, strict=True):
        column.markdown(theme.tile(label, value, note, colour, icon), unsafe_allow_html=True)

    st.markdown("")
    by_attack = metrics.get("by_attack_type") or {}
    by_severity = metrics.get("by_severity") or {}

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Attack types observed**")
            if by_attack:
                st.plotly_chart(theme.bars(by_attack), width='stretch',
                                config={"displayModeBar": False})
            else:
                st.markdown('<div class="cs-evidence">No incidents recorded yet. '
                            "Analyse an event to populate this view.</div>", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("**Severity mix**")
            if by_severity:
                st.plotly_chart(theme.donut(by_severity, "incidents"), width='stretch',
                                config={"displayModeBar": False})
                st.markdown(
                    "".join(
                        f'<span class="cs-mono" style="margin-right:.9rem">'
                        f'<span class="cs-dot" style="background:{theme.severity_colour(level)};'
                        f'margin-right:.35rem"></span>{level} {count}</span>'
                        for level, count in by_severity.items()
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="cs-evidence">Nothing to chart yet.</div>',
                            unsafe_allow_html=True)

    st.markdown("")
    incidents = fetch_incidents(limit=25)
    with st.container(border=True):
        st.markdown("**Recent incidents**")
        if incidents:
            rows = [
                [
                    row["incident_id"],
                    row["created_at"][:19].replace("T", " "),
                    row["attack_type"],
                    theme.badge(row["severity"]),
                    str(row.get("risk_score") or "—"),
                    row["approval_status"],
                    row["input_preview"][:64] + ("…" if len(row["input_preview"]) > 64 else ""),
                ]
                for row in incidents
            ]
            st.markdown(
                theme.table(
                    ["Incident", "Detected", "Attack type", "Severity", "Risk", "Approval", "Input"],
                    rows,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="cs-evidence">No incidents yet. Open Analyse incident '
                        "and submit an event.</div>", unsafe_allow_html=True)

    average = metrics.get("average_latency_seconds")
    if average is not None:
        st.caption(f"Average end-to-end analysis latency {average}s")


def page_analyse() -> None:
    theme.page_header("Analyse incident", "Submit an event for classification, grounding and risk assessment")

    with st.container(border=True):
        scenario = st.selectbox("Load a scenario", ["Blank"] + list(DEMO_SCENARIOS))
        text = st.text_area(
            "Security event",
            value=DEMO_SCENARIOS.get(scenario, ""),
            height=190,
            placeholder=(
                "Paste an alert, a log excerpt, a suspicious email, a URL, or several events "
                "separated by blank lines or 'Event N:' headers."
            ),
        )

        controls = st.columns([2, 2, 3, 2])
        use_rag = controls[0].toggle("Threat intelligence", value=True)
        set_criticality = controls[1].toggle("Asset criticality", value=False)
        criticality = controls[2].slider("Criticality", 1, 5, 3) if set_criticality else None
        analyse_clicked = controls[3].button(
            "Analyse", type="primary", disabled=not text.strip(), width='stretch'
        )

    if analyse_clicked:
        with st.spinner("running the analysis workflow…"):
            try:
                st.session_state["last_result"] = analyse(text, use_rag, criticality)
            except Exception as exc:
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
                return

    result = st.session_state.get("last_result")
    if not result:
        return

    st.markdown(
        f'<div class="cs-mono" style="margin:.9rem 0 .6rem">'
        f'{result["incident_id"]} · run {result["run_id"]} · '
        f'{" → ".join(result.get("node_path", []))}</div>',
        unsafe_allow_html=True,
    )

    if result.get("awaiting_approval"):
        render_checkpoint(result)
    elif result.get("report"):
        render_report(result["report"])

    history = result.get("history_matches") or []
    if history:
        with st.container(border=True):
            st.markdown("**Incident memory**")
            rows = [
                [
                    match["value"],
                    match["kind"],
                    str(match["incident_count"]),
                    ", ".join(item["incident_id"] for item in match.get("incidents", [])[:3]),
                ]
                for match in history
            ]
            st.markdown(
                theme.table(["Indicator", "Type", "Prior incidents", "Seen in"], rows),
                unsafe_allow_html=True,
            )

    if result.get("errors"):
        st.warning("Recorded during this run: " + "; ".join(result["errors"]))


def page_threat_intelligence() -> None:
    theme.page_header(
        "Threat intelligence",
        "Search the knowledge base the analysis pipeline uses for grounding",
    )

    with st.container(border=True):
        columns = st.columns([5, 1, 1])
        query = columns[0].text_input("Query", value="brute force authentication failures",
                                      label_visibility="collapsed")
        top_k = columns[1].number_input("Results", 1, 15, 5, label_visibility="collapsed")
        search = columns[2].button("Search", type="primary", width='stretch')

    if search and query.strip():
        with st.spinner("searching…"):
            try:
                result = search_intel(query, int(top_k))
            except Exception as exc:
                st.error(f"Search failed: {type(exc).__name__}: {exc}")
                return

        if result.get("error"):
            st.error(result["error"])
            return

        st.markdown(
            f'<div class="cs-mono" style="margin:.7rem 0">'
            f'{len(result.get("documents", []))} results · store {result.get("store")} · '
            f'{result.get("latency_seconds")}s</div>',
            unsafe_allow_html=True,
        )
        render_sources(result.get("documents", []), "Nothing matched that query.")


def page_approvals() -> None:
    theme.page_header(
        "Approval queue",
        "Incidents whose risk or recommended actions require an analyst decision",
        status="Human-in-the-loop",
        status_colour=theme.PALETTE["medium"],
    )

    try:
        pending = fetch_pending()
    except Exception as exc:
        st.error(f"Queue unavailable: {type(exc).__name__}: {exc}")
        return

    if not pending:
        st.markdown(
            '<div class="cs-card">Nothing is waiting for a decision.</div>', unsafe_allow_html=True
        )
        return

    for row in pending:
        with st.container(border=True):
            head, action = st.columns([5, 1])
            head.markdown(
                f'<span class="cs-mono" style="color:{theme.PALETTE["text"]}">{row["incident_id"]}</span> '
                f'{theme.badge(row["severity"])} '
                f'<span class="cs-mono">{row["attack_type"]} · risk {row.get("risk_score") or "—"} · '
                f'{row["created_at"][:19].replace("T", " ")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="cs-evidence">{row["input_preview"]}</div>',
                        unsafe_allow_html=True)
            if action.button("Review", key=f"review_{row['incident_id']}", width='stretch'):
                st.session_state["review_target"] = row["incident_id"]

    st.caption(
        "Open Analyse incident to act on a checkpoint, or submit the decision to "
        "POST /approval/{incident_id}."
    )


def page_methodology() -> None:
    theme.page_header("Methodology", "How this system reaches a conclusion, and what it will not do")

    with st.container(border=True):
        st.markdown(
            """
**Pipeline.** Input classification (deterministic) → threat detection (LLM) →
threat-intelligence retrieval (RAG) → correlation (deterministic, with an LLM
summary) → risk scoring (deterministic) → response recommendation (LLM, filtered)
→ human approval (checkpoint) → incident report.

**What the model decides.** The model classifies the event and proposes evidence.
It does not decide the risk score, the routing, or which threat-intelligence
identifiers are reported — those are computed in code, which is why they are
reproducible.

**Grounding rule.** A MITRE technique or CVE is reported only if it appears in
the retrieved context. Anything else the model proposes is recorded as a rejected
claim and shown in the report rather than silently dropped.
"""
        )

    st.markdown("")
    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Risk matrix**")
            st.caption(
                "Risk score = likelihood × impact. A project risk model, not a published standard."
            )
            headers = ["Likelihood ↓ / Impact →"] + [
                f"{impact} {IMPACT_SCALE[impact]}" for impact in range(1, 6)
            ]
            rows = []
            for likelihood in range(1, 6):
                cells = [f"{likelihood} {LIKELIHOOD_SCALE[likelihood]}"]
                for impact in range(1, 6):
                    score = likelihood * impact
                    band = next(
                        level.value for low, high, level in RISK_BANDS if low <= score <= high
                    )
                    colour = theme.severity_colour(band)
                    cells.append(
                        f'<span style="color:{colour};font-family:{theme.FONT_MONO}">{score}</span>'
                    )
                rows.append(cells)
            st.markdown(theme.table(headers, rows), unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("**Approval policy**")
            st.markdown(
                f"""
- Approval is required at **{get_settings().approval_severity_threshold}** risk or above.
- Approval is also required for any action that would block, isolate, disable or
  revoke something in production — at any risk level.
- Neither outcome executes anything.
"""
            )

    st.markdown("")
    with st.container(border=True):
        st.markdown("**Limitations**")
        st.markdown(
            """
- AI-assisted analysis. Every finding requires analyst validation.
- The default backend is a deterministic rules baseline; results improve with the
  fine-tuned model enabled.
- The knowledge base is a curated subset of ATT&CK, CWE and public guidance, not a
  complete threat-intelligence feed.
- Correlation produces a hypothesis from shared indicators and ordering; it is not
  proof of a single intrusion.
"""
        )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="CyberSentinel", page_icon="🛡", layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject()

    settings = get_settings()

    with st.sidebar:
        theme.brand()

        st.markdown('<div class="cs-side-label">Console</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.session_state.setdefault("api_base", settings.api_base_url)
        if "use_api" not in st.session_state:
            st.session_state["use_api"] = api_available()

    if page == "Overview":
        page_overview()
    elif page == "Analyse incident":
        page_analyse()
    elif page == "Threat intelligence":
        page_threat_intelligence()
    elif page == "Approvals":
        page_approvals()
    else:
        page_methodology()


if __name__ == "__main__":
    main()
