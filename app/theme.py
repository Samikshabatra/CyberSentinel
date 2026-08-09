"""Visual system for the CyberSentinel console.

A dark operations-centre aesthetic: near-black navy canvas, raised cards with
hairline borders, and a severity palette that is used consistently everywhere -
badges, chart series, borders and rules all draw from the same five colours, so
a colour always means the same thing.

Two typefaces carry two different jobs. Inter sets the interface. JetBrains Mono
sets anything an analyst would copy, paste or grep: incident ids, technique ids,
IP addresses, hashes. Identifiers in mono is the vernacular of security tooling,
and it makes machine-readable values visually separable from prose at a glance.

The loud element is deliberately singular: the approval checkpoint. It is the
one moment the system stops and defers to a person, so it is the one place the
interface raises its voice. Everything else stays quiet.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
PALETTE: dict[str, str] = {
    "bg": "#080B12",
    "surface": "#10151F",
    "surface_raised": "#161C28",
    "border": "#1E2635",
    "border_strong": "#2A3446",
    "text": "#E4E9F2",
    "muted": "#8792A8",
    "faint": "#5C6779",
    "brand": "#FF4D4D",
    "critical": "#F04438",
    "high": "#FF8A3D",
    "medium": "#F5C518",
    "low": "#35C46A",
    "info": "#4C9AFF",
    "unknown": "#6B7688",
}

SEVERITY_COLOURS: dict[str, str] = {
    "CRITICAL": PALETTE["critical"],
    "HIGH": PALETTE["high"],
    "MEDIUM": PALETTE["medium"],
    "LOW": PALETTE["low"],
    "UNKNOWN": PALETTE["unknown"],
}

#: Ordered series colours for charts, so a category keeps its colour everywhere.
CHART_SEQUENCE: list[str] = [
    PALETTE["critical"],
    PALETTE["high"],
    PALETTE["medium"],
    PALETTE["info"],
    PALETTE["low"],
    "#A78BFA",
    "#22D3EE",
    "#F472B6",
]

FONT_UI = "'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', 'Cascadia Mono', Consolas, monospace"


def severity_colour(level: str | None) -> str:
    return SEVERITY_COLOURS.get(str(level or "UNKNOWN").upper(), PALETTE["unknown"])


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
def _css() -> str:
    p = PALETTE
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --bg: {p["bg"]};
  --surface: {p["surface"]};
  --surface-raised: {p["surface_raised"]};
  --border: {p["border"]};
  --border-strong: {p["border_strong"]};
  --text: {p["text"]};
  --muted: {p["muted"]};
  --faint: {p["faint"]};
  --brand: {p["brand"]};
  --critical: {p["critical"]};
  --high: {p["high"]};
  --medium: {p["medium"]};
  --low: {p["low"]};
  --info: {p["info"]};
}}

/* ---------- canvas ---------- */
.stApp {{
  background: var(--bg);
  color: var(--text);
  font-family: {FONT_UI};
}}
.block-container {{ padding: 1.5rem 2rem 4rem; max-width: 1600px; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

h1, h2, h3, h4 {{ color: var(--text); font-family: {FONT_UI}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.05rem !important; font-weight: 600; }}
h3 {{ font-size: .92rem !important; font-weight: 600; }}
p, li, span, label {{ color: var(--text); }}
.block-container p {{ font-size: .86rem; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: #0B0F18;
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.25rem; }}

.cs-brand {{ display: flex; align-items: center; gap: .7rem; padding: .2rem .25rem 1.1rem; }}
.cs-brand-mark {{
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(145deg, var(--brand), #C2261F);
  display: grid; place-items: center; font-size: 1.05rem;
  box-shadow: 0 0 0 1px rgba(255,77,77,.25), 0 6px 16px -8px rgba(255,77,77,.8);
}}
.cs-brand-name {{ font-size: 1rem; font-weight: 650; line-height: 1.15; }}
.cs-brand-sub {{
  font-family: {FONT_MONO}; font-size: .62rem; color: var(--faint);
  text-transform: uppercase; letter-spacing: .09em;
}}

/* sidebar navigation, restyled from the radio group */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: .15rem; display: flex; flex-direction: column; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  padding: .5rem .7rem; border-radius: 8px; cursor: pointer;
  border: 1px solid transparent; transition: background .13s ease, border-color .13s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,.035); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
  font-size: .84rem; color: var(--muted); font-weight: 500;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background: rgba(76,154,255,.10); border-color: rgba(76,154,255,.28);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color: var(--text); font-weight: 600; }}
/* Hide the radio dial - selection is carried by the row treatment, and a
   navigation list should not look like a form. The dial is a rounded div nested
   three levels inside the label, not the label's own span. */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div > div:first-child {{
  display: none !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > span:first-child {{ display: none !important; }}
section[data-testid="stSidebar"] div[role="radiogroup"] input {{ display: none; }}

.cs-side-label {{
  font-family: {FONT_MONO}; font-size: .6rem; color: var(--faint);
  text-transform: uppercase; letter-spacing: .11em; margin: 1.1rem 0 .45rem .3rem;
}}
.cs-status {{
  border: 1px solid var(--border); background: var(--surface);
  border-radius: 10px; padding: .7rem .8rem; margin-top: .4rem;
}}
.cs-status-row {{
  display: flex; justify-content: space-between; align-items: center;
  font-size: .74rem; padding: .16rem 0;
}}
.cs-status-key {{ color: var(--faint); }}
.cs-status-val {{ font-family: {FONT_MONO}; color: var(--muted); font-size: .7rem; }}

/* ---------- page header ---------- */
.cs-header {{
  display: flex; justify-content: space-between; align-items: flex-start;
  padding: 0 0 .85rem; margin-bottom: 1.1rem; border-bottom: 1px solid var(--border);
}}
/* The page title is a styled div, not an <h1>: Streamlit lifts real headings out
   of custom markup into their own anchor container, which breaks the layout and
   escapes these rules. */
.cs-title {{
  font-size: 1.35rem; font-weight: 650; line-height: 1.2;
  color: var(--text); letter-spacing: -0.015em; margin-bottom: .25rem;
}}
.cs-header-sub {{ color: var(--muted); font-size: .82rem; }}
.cs-pill {{
  display: inline-flex; align-items: center; gap: .42rem;
  border: 1px solid var(--border-strong); border-radius: 999px;
  padding: .32rem .75rem; font-size: .74rem; color: var(--muted); background: var(--surface);
  white-space: nowrap;
}}
.cs-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
@keyframes cs-pulse {{ 0%,100% {{ opacity: 1 }} 50% {{ opacity: .35 }} }}
.cs-dot-live {{ animation: cs-pulse 2.4s ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{ .cs-dot-live {{ animation: none; }} }}

/* ---------- cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 12px; padding: 1rem 1.1rem;
}}
.cs-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
}}
.cs-card-title {{
  font-size: .82rem; font-weight: 600; color: var(--text);
  margin-bottom: .8rem; display: flex; justify-content: space-between; align-items: center;
}}

/* ---------- metric tiles ---------- */
.cs-tile {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: .9rem 1rem; position: relative; overflow: hidden; height: 100%;
}}
.cs-tile::before {{
  content: ''; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--accent, var(--info));
}}
.cs-tile-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .45rem; }}
.cs-tile-label {{ font-size: .74rem; color: var(--muted); font-weight: 500; }}
.cs-tile-icon {{ font-size: .82rem; opacity: .85; }}
.cs-tile-value {{ font-size: 1.75rem; font-weight: 680; line-height: 1.05; color: var(--accent, var(--text)); }}
.cs-tile-note {{ font-size: .7rem; color: var(--faint); margin-top: .3rem; }}

/* ---------- badges ---------- */
.cs-badge {{
  display: inline-block; padding: .16rem .5rem; border-radius: 5px;
  font-family: {FONT_MONO}; font-size: .64rem; font-weight: 600; letter-spacing: .05em;
}}
.cs-tag {{
  display: inline-block; padding: .18rem .5rem; margin: 0 .28rem .3rem 0;
  border-radius: 5px; border: 1px solid var(--border-strong); background: var(--surface-raised);
  font-family: {FONT_MONO}; font-size: .68rem; color: var(--info);
}}
.cs-mono {{ font-family: {FONT_MONO}; font-size: .78rem; color: var(--muted); }}

/* ---------- data table ---------- */
.cs-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; }}
.cs-table th {{
  text-align: left; padding: .5rem .6rem; color: var(--faint);
  font-family: {FONT_MONO}; font-size: .63rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .07em;
  border-bottom: 1px solid var(--border);
}}
.cs-table td {{ padding: .55rem .6rem; border-bottom: 1px solid rgba(30,38,53,.55); color: var(--muted); }}
.cs-table tr:last-child td {{ border-bottom: none; }}
.cs-table tr:hover td {{ background: rgba(255,255,255,.018); }}
.cs-table td.cs-id {{ font-family: {FONT_MONO}; font-size: .72rem; color: var(--text); }}

/* ---------- evidence and sources ---------- */
.cs-evidence {{
  border-left: 2px solid var(--border-strong); padding: .1rem 0 .1rem .75rem;
  margin-bottom: .5rem; font-size: .82rem; color: var(--muted);
}}
.cs-source {{
  border: 1px solid var(--border); border-radius: 9px; padding: .6rem .75rem;
  margin-bottom: .5rem; background: var(--surface-raised);
}}
.cs-source-head {{ display: flex; justify-content: space-between; gap: .75rem; align-items: baseline; }}
.cs-source-id {{ font-family: {FONT_MONO}; font-size: .72rem; color: var(--info); }}
.cs-source-title {{ font-size: .8rem; color: var(--text); }}
.cs-source-meta {{ font-family: {FONT_MONO}; font-size: .64rem; color: var(--faint); }}

/* ---------- the signature: approval checkpoint ---------- */
.cs-hold {{
  border: 1px solid rgba(255,138,61,.38);
  border-left: 3px solid var(--high);
  border-radius: 12px;
  background:
    linear-gradient(180deg, rgba(255,138,61,.09), rgba(255,138,61,0) 55%),
    var(--surface);
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}}
.cs-hold-eyebrow {{
  font-family: {FONT_MONO}; font-size: .66rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .16em; color: var(--high);
  display: flex; align-items: center; gap: .5rem; margin-bottom: .45rem;
}}
.cs-hold-reason {{ font-size: .92rem; color: var(--text); margin-bottom: .2rem; }}
.cs-hold-note {{ font-size: .76rem; color: var(--muted); }}
.cs-impact {{
  border: 1px solid rgba(240,68,56,.35); border-radius: 9px;
  background: rgba(240,68,56,.07); padding: .7rem .85rem; margin-top: .7rem;
}}
.cs-impact-label {{
  font-family: {FONT_MONO}; font-size: .63rem; text-transform: uppercase;
  letter-spacing: .1em; color: var(--critical); margin-bottom: .4rem;
}}
.cs-impact-item {{ font-size: .84rem; color: var(--text); padding: .12rem 0; }}

/* ---------- controls ---------- */
.stButton > button {{
  background: var(--surface-raised); color: var(--text);
  border: 1px solid var(--border-strong); border-radius: 8px;
  font-size: .82rem; font-weight: 550; padding: .42rem 1rem;
  transition: border-color .13s ease, background .13s ease;
}}
.stButton > button:hover {{ border-color: var(--info); background: rgba(76,154,255,.10); color: var(--text); }}
.stButton > button:focus-visible {{ outline: 2px solid var(--info); outline-offset: 2px; }}
.stButton > button[kind="primary"] {{
  background: var(--info); border-color: var(--info); color: #07101F; font-weight: 620;
}}
.stButton > button[kind="primary"]:hover {{ background: #6BADFF; border-color: #6BADFF; color: #07101F; }}

.stTextArea textarea, .stTextInput input {{
  background: #0C1119 !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 9px !important;
  font-size: .84rem !important;
}}
.stTextArea textarea {{ font-family: {FONT_MONO} !important; font-size: .8rem !important; }}
.stTextArea textarea:focus, .stTextInput input:focus {{ border-color: var(--info) !important; }}

div[data-baseweb="select"] > div {{
  background: #0C1119 !important; border-color: var(--border) !important; border-radius: 9px !important;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: .2rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
  background: transparent; color: var(--muted); font-size: .81rem;
  padding: .5rem .85rem; border-radius: 7px 7px 0 0;
}}
.stTabs [aria-selected="true"] {{ color: var(--text); border-bottom: 2px solid var(--info); }}

div[data-testid="stExpander"] details {{
  background: var(--surface-raised); border: 1px solid var(--border); border-radius: 9px;
}}
div[data-testid="stExpander"] summary {{ font-size: .78rem; color: var(--muted); }}

div[data-testid="stAlert"] {{ border-radius: 9px; border: 1px solid var(--border); font-size: .84rem; }}
hr {{ border-color: var(--border); }}
code {{ font-family: {FONT_MONO}; background: var(--surface-raised); color: var(--info); font-size: .78rem; }}

/* ---------- accessibility ---------- */
*:focus-visible {{ outline: 2px solid var(--info); outline-offset: 2px; }}
@media (max-width: 900px) {{ .block-container {{ padding: 1rem; }} }}
</style>
"""


def inject() -> None:
    """Apply the stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def brand() -> None:
    """Product mark for the sidebar."""
    st.markdown(
        """
        <div class="cs-brand">
          <div class="cs-brand-mark">🛡</div>
          <div>
            <div class="cs-brand-name">CyberSentinel</div>
            <div class="cs-brand-sub">Threat Analysis Console</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, status: str = "", status_colour: str = "") -> None:
    """Page title with an optional status pill on the right."""
    pill = ""
    if status:
        colour = status_colour or PALETTE["low"]
        pill = (
            f'<div class="cs-pill">'
            f'<span class="cs-dot cs-dot-live" style="background:{colour}"></span>{status}</div>'
        )
    st.markdown(
        f'<div class="cs-header"><div><div class="cs-title">{title}</div>'
        f'<div class="cs-header-sub">{subtitle}</div></div>{pill}</div>',
        unsafe_allow_html=True,
    )


def tile(label: str, value: Any, note: str = "", colour: str = "", icon: str = "") -> str:
    """A metric tile. Returns HTML so tiles can be placed in columns."""
    accent = colour or PALETTE["info"]
    icon_html = f'<span class="cs-tile-icon">{icon}</span>' if icon else ""
    note_html = f'<div class="cs-tile-note">{note}</div>' if note else ""
    return (
        f'<div class="cs-tile" style="--accent:{accent}">'
        f'<div class="cs-tile-head"><span class="cs-tile-label">{label}</span>{icon_html}</div>'
        f'<div class="cs-tile-value">{value}</div>{note_html}</div>'
    )


def badge(level: str) -> str:
    """Severity badge; colour is the same everywhere the level appears."""
    text = str(level or "UNKNOWN").upper()
    colour = severity_colour(text)
    return (
        f'<span class="cs-badge" style="background:{colour}1F;color:{colour};'
        f'border:1px solid {colour}44">{text}</span>'
    )


def tag(value: str) -> str:
    """Monospace chip for an identifier."""
    return f'<span class="cs-tag">{value}</span>'


def card(title: str, body_html: str, aside: str = "") -> str:
    aside_html = f'<span class="cs-mono">{aside}</span>' if aside else ""
    return (
        f'<div class="cs-card"><div class="cs-card-title"><span>{title}</span>'
        f"{aside_html}</div>{body_html}</div>"
    )


def table(headers: list[str], rows: list[list[str]], id_column: int = 0) -> str:
    """Compact data table. `id_column` is rendered in mono as an identifier."""
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="cs-id">{cell}</td>' if index == id_column else f"<td>{cell}</td>"
            for index, cell in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    return f'<table class="cs-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def evidence_list(items: list[str]) -> str:
    if not items:
        return '<div class="cs-evidence">No supporting evidence was extracted.</div>'
    return "".join(f'<div class="cs-evidence">{item}</div>' for item in items)


def status_row(key: str, value: str) -> str:
    return (
        f'<div class="cs-status-row"><span class="cs-status-key">{key}</span>'
        f'<span class="cs-status-val">{value}</span></div>'
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _layout(height: int, margin: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "height": height,
        "margin": margin or {"l": 8, "r": 8, "t": 8, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": PALETTE["muted"], "family": "Inter, sans-serif", "size": 11},
        "showlegend": False,
    }


def donut(values: dict[str, int], centre_label: str, height: int = 210) -> Any:
    """Distribution donut with the total in the middle."""
    import plotly.graph_objects as go

    labels = list(values)
    colours = [
        SEVERITY_COLOURS.get(label.upper(), CHART_SEQUENCE[index % len(CHART_SEQUENCE)])
        for index, label in enumerate(labels)
    ]
    total = sum(values.values())

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=list(values.values()),
            hole=0.68,
            marker={"colors": colours, "line": {"color": PALETTE["surface"], "width": 2}},
            textinfo="none",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout(height),
        annotations=[
            {
                "text": f"<b>{total}</b><br><span style='font-size:10px'>{centre_label}</span>",
                "showarrow": False,
                "font": {"size": 22, "color": PALETTE["text"]},
            }
        ],
    )
    return figure


def bars(values: dict[str, int], height: int = 230) -> Any:
    """Horizontal ranked bars, highest at the top."""
    import plotly.graph_objects as go

    ordered = sorted(values.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    counts = [item[1] for item in ordered]
    colours = [
        SEVERITY_COLOURS.get(label.upper(), CHART_SEQUENCE[index % len(CHART_SEQUENCE)])
        for index, label in enumerate(labels)
    ]

    figure = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker={"color": colours, "line": {"width": 0}},
            # Fixed thickness: with only two or three categories Plotly would
            # otherwise render slabs that dominate the card.
            width=0.5,
            text=counts,
            textposition="outside",
            textfont={"color": PALETTE["muted"], "size": 10},
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    figure.update_layout(**_layout(height), bargap=0.45)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(showgrid=False, tickfont={"size": 10.5, "color": PALETTE["muted"]})
    return figure


def risk_gauge(score: int, level: str, height: int = 150) -> Any:
    """Risk score on the 1-25 scale, coloured by band."""
    import plotly.graph_objects as go

    colour = severity_colour(level)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 30, "color": colour}},
            gauge={
                "axis": {
                    "range": [0, 25],
                    "tickvals": [0, 4, 9, 16, 25],
                    "tickfont": {"size": 9, "color": PALETTE["faint"]},
                },
                "bar": {"color": colour, "thickness": 0.7},
                "bgcolor": PALETTE["surface_raised"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 4], "color": "rgba(53,196,106,.13)"},
                    {"range": [4, 9], "color": "rgba(245,197,24,.13)"},
                    {"range": [9, 16], "color": "rgba(255,138,61,.13)"},
                    {"range": [16, 25], "color": "rgba(240,68,56,.13)"},
                ],
            },
        )
    )
    figure.update_layout(**_layout(height, {"l": 22, "r": 22, "t": 12, "b": 4}))
    return figure
