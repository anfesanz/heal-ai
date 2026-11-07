from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_starts_without_exceptions() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "HEAL-AI Eval"
    assert len(app.selectbox) == 2
