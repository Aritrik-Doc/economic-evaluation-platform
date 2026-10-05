from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_streamlit_route_executes_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20)
    app.run()
    assert len(app.exception) == 0
