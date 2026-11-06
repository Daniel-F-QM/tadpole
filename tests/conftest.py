"""Shared fixtures. GUI tests run offscreen; the Julia engine is started once
per session (it takes ~15 s to boot) and only for tests that ask for it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    from tadpole.design.theme import install_theme
    from tadpole.gui.widgets import use_dot_decimals
    app = QApplication.instance() or QApplication([])
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
