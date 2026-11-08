"""The window builds, prepares a simulated trace, and switches theme —
without starting Julia."""

import numpy as np
import pytest

CLIENT_1080P_150 = (1280, 720 - 48 - 31)


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


# ── layout, Analysis page, plot interaction ────────────────────────────────
def _fake_result(p, n=50):
    E = np.exp(-(p.t_fs / 30) ** 2).astype(complex)
    errs = list(np.linspace(5e-2, 1e-2, n))
    return dict(trace=p.trace * 0.95, E=E, errors=errs, errors_p=[e * 5 for e in errs],
                G=errs[-1], Gp=errs[-1] * 5, iterations=n, elapsed_s=1.0,
                stopped="max_iter", algorithm="pcgpa")


def test_control_columns_fit_without_scrolling(win, qapp):
    # the client area of a maximized window on a 1280×720 logical screen
    # (1080p at 150 %) with a 48 px taskbar and a 31 px title bar
    win.resize(*CLIENT_1080P_150)
    win.prep.simulate()
    win.prep.send()
    win.propagator.wait_ready()
    for i, col in enumerate((win.prep_column, win.recon_column, win.analysis_column,
                             win.propagator_column)):
        win.set_mode(i)
        qapp.processEvents()
        assert not col.needs_scrolling(), (i, col.inner.minimumSizeHint().height(),
                                           col.viewport().height())
        assert not col.verticalScrollBar().isVisible(), i
        # …and nothing wider than the panel (it would scroll sideways)
        assert not col.too_wide(), (i, col.inner.layout().minimumSize().width())
        # room to spare, so a longer file name or warning does not tip it over
        assert col.inner.minimumSizeHint().height() <= col.viewport().height() - 30, i


def test_analysis_page_after_a_result(win, qapp):
    from tadpole.engine.protocol import Done
    win.prep.simulate()
    win.prep.send()
    assert not win.btn_mode_analysis.isEnabled()
    win.recon._on_done(Done(job_id=1, result=_fake_result(win.state.recon_input)))
    assert win.btn_mode_analysis.isEnabled()
    assert win.analysis_canvas.result is not None
    assert "diff_cbar" in win.analysis_canvas._cbars
    win.set_mode(2)
    assert win.current_canvas() is win.analysis_canvas
    # new input: the old analysis no longer applies
    win.recon.set_input(win.state.prepared)
    assert not win.btn_mode_analysis.isEnabled()
    assert win.stack.currentIndex() == 1


def test_measured_and_retrieved_share_one_colour_range(win, qapp):
    win.prep.simulate()
    win.prep.send()
    c = win.recon_canvas
    ims = c._cbars["trace"]["images"]
    assert len(ims) == 2
    c.set_clim("trace", 0.1, 0.5)
    assert all(im.get_clim() == (0.1, 0.5) for im in ims)
    c.redraw()                                    # survives a rebuild…
    assert all(im.get_clim() == (0.1, 0.5) for im in c._cbars["trace"]["images"])
    c.reset_view("trace")                         # …until right-click
    assert c._cbars["trace"]["images"][0].get_clim() == (0.0, 1.0)


def test_zoom_survives_redraw_and_resets(win, qapp):
    win.prep.simulate()
    win.prep.send()
    c = win.recon_canvas
    home = c._axes["meas"].get_xlim()
    c._apply_zoom(c._axes["meas"], -20, 20, 4.6, 4.8)
    assert c._axes["ret"].get_xlim() == (-20, 20)      # shared with Retrieved
    win.appearance.toggle_theme()                       # rebuilds the figure
    win.appearance.toggle_theme()
    assert c._axes["meas"].get_xlim() == (-20, 20)
    assert c._axes["ret"].get_ylim() == (4.6, 4.8)
    c.reset_view("ret")
    assert c._axes["meas"].get_xlim() == home


def test_typed_axis_bound(win, qapp):
    win.prep.simulate()
    c = win.prep_canvas
    ax = c._views["grid"]["ax"]
    lo, hi = ax.get_xlim()
    c.apply_axis_bound(ax, "x", "hi", hi / 2)
    assert ax.get_xlim() == (lo, hi / 2)
    assert c.is_manual("grid")
    msgs = []
    c.message.connect(msgs.append)
    c.apply_axis_bound(ax, "x", "lo", hi)               # would invert: refused
    assert ax.get_xlim() == (lo, hi / 2) and msgs


def test_no_black_text_on_dark_plots(win, qapp):
    from matplotlib.colors import to_rgb
    from tadpole.engine.protocol import Done
    win.prep.simulate()
    win.prep.send()
    win.recon._on_done(Done(job_id=1, result=_fake_result(win.state.recon_input)))
    for c in win.canvases():
        c.draw()
        for ax in c.fig.axes + [a for ax in c.fig.axes for a in ax.child_axes]:
            texts = [t for axis in (ax.xaxis, ax.yaxis)
                     for t in axis.get_ticklabels(which="both") + [axis.label,
                                                                   axis.get_offset_text()]]
            for t in texts:
                if t.get_visible() and t.get_text():
                    assert sum(to_rgb(t.get_color())) > 0.6, (c, t.get_text())


def test_trace_view_is_centred(win, qapp):
    from tadpole.analysis import centred_range
    x = np.linspace(-300, 300, 601)
    y = np.exp(-(x / 30) ** 2) + 0.01 * (x < -250)     # lopsided noise on one side
    lo, hi = centred_range(x, y, 0.0)
    assert lo == pytest.approx(-hi)
    win.prep.simulate()
    win.prep.send()
    c, p = win.recon_canvas, win.state.recon_input
    lo, hi = c._axes["meas"].get_xlim()
    assert lo == pytest.approx(-hi)
    lo, hi = c._axes["meas"].get_ylim()
    assert (lo + hi) / 2 == pytest.approx(p.w0_rad_fs)



# ── Propagator ──────────────────────────────────────────────────────────────
def _measured_from(E, phase, p):
    from tadpole.prep.model import PreparedTrace
    from tadpole.propagation.physics import propagate, shg_trace
    T = shg_trace(propagate(E, phase))
    return PreparedTrace(trace=T, mask=np.ones_like(T, bool), dt_fs=p.dt_fs,
                         w0_rad_fs=p.w0_rad_fs, trace_meas=T, mask_meas=np.ones_like(T, bool),
                         delays_meas_fs=p.t_fs, source="synthetic")


@pytest.mark.parametrize("reverse", [False, True])
def test_propagator_finds_the_direction(win, qapp, reverse):
    from tadpole.engine.protocol import Done
    from tadpole.propagation.physics import propagate, time_reverse
    ctl = win.propagator
    ctl.wait_ready()
    win.prep.simulate()
    win.prep.send()
    p = win.state.recon_input
    w = p.w_rad_fs
    E = propagate(np.exp(-(p.t_fs / 25) ** 2).astype(complex), -400 / 2 * w ** 2)
    res = _fake_result(p)
    res["E"] = E
    win.recon._on_done(Done(job_id=1, result=res))
    assert ctl.pulse is not None and ctl.pulse.n == p.n
    ctl.set_stack([{"shelf": "main", "book": "SiO2", "page": "Malitson",
                    "thickness_mm": 4.0}])
    ctl.recompute()
    assert ctl.view is not None and ctl.view.gdd == pytest.approx(4 * 36.16, rel=0.05)
    truth = time_reverse(E) if reverse else E
    ctl.set_measured_source("npz", _measured_from(truth, ctl._phase, p))
    ctl.recompute()
    ctl.wait_scan()
    assert ("time-reversed" if reverse else "as retrieved") in win.lbl_verdict.text()
    assert ctl.view.scan[2] == pytest.approx(-1 if reverse else 1, abs=0.1)
    # with every readout filled in, the panel still fits a 1080p/150 % screen
    win.resize(*CLIENT_1080P_150)
    win.set_mode(3)
    qapp.processEvents()
    assert not win.propagator_column.needs_scrolling()
    assert not win.propagator_column.too_wide()


def test_propagator_stack_editing(win, qapp):
    ctl = win.propagator
    ctl.wait_ready()
    n = len(ctl.stack)
    ctl.add_element()
    assert len(ctl.stack) == n + 1 == win.list_stack.count()
    win.list_stack.setCurrentRow(n)
    i = win.cmb_material.findData("main/CaF2")
    assert i >= 0
    win.cmb_material.setCurrentIndex(i)
    ctl._material_chosen(i)
    assert ctl.stack[n].book == "CaF2" and win.cmb_page.count() > 0
    win.sp_thickness.setValue(3.5)
    assert ctl.stack[n].thickness_mm == 3.5
    assert "CaF2" in win.list_stack.item(n).text()
    snap = win.settings_ctl.snapshot()
    assert snap["prop_stack"][n]["book"] == "CaF2"
    ctl.remove_element()
    assert len(ctl.stack) == n


def test_plots_do_not_move_when_numbers_change(win, qapp):
    """The fixed layout: panel positions depend on the window size only, not
    on the tick labels (constrained layout used to shift them every redraw)."""
    from tadpole.engine.protocol import Done
    win.prep.simulate()
    win.prep.send()
    p = win.state.recon_input
    c = win.recon_canvas

    def positions(canvas):
        canvas.draw()
        return [tuple(np.round(a.get_position().bounds, 6)) for a in canvas.fig.axes]

    E1 = np.exp(-(p.t_fs / 30) ** 2).astype(complex)
    c.show_progress(1, 1e-2, 5e-2, p.trace * 0.9, E1)
    before = positions(c)
    # a much narrower pulse with a big phase: very different tick labels
    E2 = (np.exp(-(p.t_fs / 8) ** 2) * np.exp(1j * 0.05 * p.t_fs ** 2)).astype(complex)
    c.show_progress(2, 1e-3, 5e-3, p.trace * 0.5, E2)
    assert positions(c) == before
    c.show_result(_fake_result(p))
    assert positions(c) == before

    pc = win.propagator_canvas
    ctl = win.propagator
    ctl.wait_ready()
    win.recon._on_done(Done(job_id=1, result=_fake_result(p)))
    ctl.set_stack([{"shelf": "main", "book": "SiO2", "page": "Malitson", "thickness_mm": 1.0}])
    ctl.recompute()
    before = positions(pc)
    ctl.set_stack([{"shelf": "main", "book": "SiO2", "page": "Malitson", "thickness_mm": 25.0}])
    ctl.recompute()
    assert positions(pc) == before
