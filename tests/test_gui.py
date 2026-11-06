"""The window builds, prepares a simulated trace, and switches theme —
without starting Julia."""

import numpy as np
import pytest


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    import tadpole.settings as settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "settings.json")
    settings._reset_cache()
    from tadpole.gui.window.main_window import TadpoleWindow
    w = TadpoleWindow("dark", start_engine=False)
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.recon.shutdown()
    w.hide()
    settings._reset_cache()


def test_simulate_prepare_send(win, qapp):
    win.prep.simulate()
    assert win.state.prepared is not None
    assert win.btn_send.isEnabled()
    win.prep.send()
    qapp.processEvents()
    assert win.stack.currentIndex() == 1
    assert win.state.recon_input is win.state.prepared
    # the engine is off, so Start stays disabled
    assert not win.btn_start.isEnabled()


def test_settings_change_recomputes(win, qapp):
    win.prep.simulate()
    win.cmb_n.setCurrentIndex(win.cmb_n.findData(64))
    win.prep.recompute()
    assert win.state.prepared.n == 64


def test_theme_toggle_and_snapshot(win, qapp):
    from tadpole.design.tokens import DARK_PALETTE, LIGHT_PALETTE, PALETTE
    win.prep.simulate()
    pal = PALETTE
    win.appearance.toggle_theme()
    assert PALETTE is pal and PALETTE["bg"] == LIGHT_PALETTE["bg"]
    win.appearance.toggle_theme()
    assert PALETTE["bg"] == DARK_PALETTE["bg"]
    snap = win.settings_ctl.snapshot()
    assert snap["grid_n"] == 128 and snap["algorithm"] == "pcgpa"


def test_unimplemented_algorithm_cannot_start(win, qapp):
    win.prep.simulate()
    win.prep.send()
    win.recon.engine.state = "ready"          # pretend Julia is up
    win.cmb_alg.setCurrentIndex(win.cmb_alg.findData("pcgpa"))
    win.recon._update_buttons()
    assert win.btn_start.isEnabled()
    win.cmb_alg.setCurrentIndex(win.cmb_alg.findData("ptychographic"))
    assert not win.btn_start.isEnabled()
    win.recon.engine.state = "off"


def test_no_engine_mode(win, qapp, monkeypatch):
    monkeypatch.setenv("TADPOLE_NO_ENGINE", "1")
    win.recon.start_engine()
    assert win.recon.engine.state == "off"
    assert "disabled" in win.lbl_engine.text()
    win.prep.simulate()                      # Prepare still works
    assert win.state.prepared is not None


def test_progress_display(win, qapp):
    win.prep.simulate()
    win.prep.send()
    p = win.state.recon_input
    E = np.exp(-(p.t_fs / 30) ** 2).astype(complex)
    win.recon_canvas.show_progress(1, 1e-2, 5e-2, p.trace * 0.9, E)
    win.recon_canvas.show_progress(2, 5e-3, 3e-2, p.trace * 0.95, E)
    assert win.recon_canvas.G == [1e-2, 5e-3]
