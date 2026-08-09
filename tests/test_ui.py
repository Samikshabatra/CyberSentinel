"""Streamlit UI smoke tests.

These run the dashboard script in-process with Streamlit's app-testing harness,
so a rendering error fails the suite rather than appearing only in a browser.
"""

from __future__ import annotations

import pytest

from cybersentinel.utils.config import PROJECT_ROOT

APP_PATH = str(PROJECT_ROOT / "app" / "streamlit_app.py")

pytest.importorskip("streamlit", reason="the ui extra is not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402


def click(app: AppTest, label: str) -> AppTest:
    """Click a button by its label rather than by position.

    The sidebar contributes its own buttons, so index-based lookup is fragile.
    """
    for button in app.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"button {label!r} not found; available: {[b.label for b in app.button]}")


def run_app(page: str | None = None, timeout: float = 90.0) -> AppTest:
    """Run the dashboard, optionally selecting a page first."""
    app = AppTest.from_file(APP_PATH, default_timeout=timeout)
    # Force the in-process path so the tests never depend on a running API.
    app.session_state["use_api"] = False
    app.run()

    if page:
        app.sidebar.radio[0].set_value(page).run()

    return app


def test_app_starts_without_exception():
    app = run_app()
    assert not app.exception
    # The product mark is rendered as themed markup in the sidebar rather than
    # st.title, so assert on the sidebar content.
    sidebar_text = " ".join(element.value for element in app.sidebar.markdown)
    assert "CyberSentinel" in sidebar_text


@pytest.mark.parametrize(
    "page",
    ["Overview", "Analyse incident", "Threat intelligence", "Approvals", "Methodology"],
)
def test_every_page_renders(page):
    app = run_app(page)
    assert not app.exception, f"{page} raised: {app.exception}"


def test_analysis_flow_reaches_the_approval_checkpoint():
    app = run_app("Analyse incident")

    app.text_area[0].set_value(
        "47 failed SSH login attempts from 198.51.100.23 within 3 minutes for user root."
    )
    app = click(app, "Analyse")

    assert not app.exception
    assert "last_result" in app.session_state
    result = app.session_state["last_result"]
    assert result["awaiting_approval"]
    assert result["pending_approval"]["risk_level"] in ("HIGH", "CRITICAL")

    # The checkpoint must offer exactly the three analyst decisions.
    labels = {button.label for button in app.button}
    assert {"Approve", "Reject", "Escalate"} <= labels


def test_benign_analysis_renders_a_report():
    app = run_app("Analyse incident")

    app.text_area[0].set_value(
        "User alice authenticated successfully to the VPN at 08:52 from the usual office "
        "address, matching their normal weekday pattern. Routine activity."
    )
    app = click(app, "Analyse")

    assert not app.exception
    result = app.session_state["last_result"]
    assert not result["awaiting_approval"]
    assert result["report"]["attack_type"] == "Benign"


def test_methodology_page_documents_the_risk_matrix():
    app = run_app("Methodology")
    text = " ".join(
        [element.value for element in app.markdown] + [element.value for element in app.caption]
    ).lower()

    assert "likelihood" in text
    assert "analyst validation" in text
    assert "grounding" in text
