"""Shared fixtures. GUI tests run offscreen; the Julia engine is started once
per session (it takes ~15 s to boot) and only for tests that ask for it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Offscreen Qt finds no fonts on Windows by itself and draws every glyph as a
# wide box, which makes the layout tests measure nonsense widths.
_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
if os.path.isdir(_fonts):
    os.environ.setdefault("QT_QPA_FONTDIR", _fonts)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication, QStyleFactory
    from tadpole.design.theme import install_theme
    from tadpole.gui.widgets import use_dot_decimals
    app = QApplication.instance() or QApplication([])
    # Offscreen Qt defaults to Fusion, the real app on Windows to windows11,
    # whose buttons and boxes are taller: measure layouts in the real one.
    if "windows11" in [k.lower() for k in QStyleFactory.keys()]:
        app.setStyle("windows11")
    use_dot_decimals()
    install_theme(app, "dark")
    yield app


@pytest.fixture(scope="session")
def engine():
    """A running EngineClient, shared by every engine test."""
    import time
    from tadpole.engine.client import EngineClient
    c = EngineClient()
    c.start()
    t0 = time.monotonic()
    while c.state == "starting" and time.monotonic() - t0 < 240:
        c.poll()
        time.sleep(0.05)
    if c.state != "ready":
        c.shutdown()
        pytest.skip(f"Julia engine not available: {c.last_error or c.state}")
    yield c
    c.shutdown()
