"""The Propagator's physics: material data, sign conventions, the SHG model
and the direction verdict."""

import numpy as np
import pytest

from tadpole.analysis import t2w, w2t
from tadpole.propagation import materials as M
from tadpole.propagation.physics import (C_NM_PER_FS, best_s, propagate,
                                         pulse_dispersion, scan, shg_trace,
                                         stack_phase, taylor, time_reverse,
                                         trace_errors, verdict)

W800 = 2 * np.pi * C_NM_PER_FS / 800.0


def grid(n=256, dt=1.0):
    t = (np.arange(n) - n // 2) * dt
    w = (np.arange(n) - n // 2) * 2 * np.pi / (n * dt)
    return t, w


def chirped(t, w, fwhm=20.0, gdd=0.0, tod=0.0):
    """A Gaussian with GDD/TOD in the material sense (positive = like glass)."""
    E = np.exp(-2 * np.log(2) * (t / fwhm) ** 2).astype(complex)
    return propagate(E, gdd / 2 * w ** 2 + tod / 6 * w ** 3)


def test_taylor_of_a_cubic():
    c = taylor(lambda w: 3 + 2 * (w - 2) + 5 / 2 * (w - 2) ** 2 + 7 / 6 * (w - 2) ** 3, 2.0)
    assert c == pytest.approx([3, 2, 5, 7], rel=1e-6)


def test_fused_silica_gdd_matches_literature():
    el = M.Element("main", "SiO2", "Malitson", 1.0)
    sp = stack_phase([(M.index_function(el), 1.0)], np.array([W800]), W800)
    assert sp.gdd == pytest.approx(36.16, abs=0.3)       # fs²/mm at 800 nm
    assert sp.tod == pytest.approx(27.5, abs=1.0)        # fs³/mm


def test_index_is_nan_outside_the_data_range():
    el = M.Element("main", "SiO2", "Malitson", 1.0)
    lo, hi = M.wl_range(el)
    n = M.index_function(el)(np.array([lo / 2, 800.0, hi * 2]))
    assert np.isnan(n[0]) and np.isnan(n[2]) and 1.4 < n[1] < 1.5


def test_missing_data_is_taylor_filled_and_flagged():
    t, w = grid()
    n_of = lambda wl: np.where(np.abs(wl - 800) < 50, 1.5 + 1e4 / wl ** 2, np.nan)  # noqa
    sp = stack_phase([(n_of, 1.0)], W800 / 2 * 0 + W800 + w * 0.1, W800)
    assert np.all(np.isfinite(sp.total))
    assert sp.elements[0].filled.any() and not sp.elements[0].filled.all()


def test_positive_gdd_delays_blue_behind_red():
    t, w = grid(512)
    Ew = t2w(chirped(t, w, gdd=500.0))
    centroid = lambda f: float((t * np.abs(f) ** 2).sum() / (np.abs(f) ** 2).sum())  # noqa
    assert centroid(w2t(Ew * (w < 0))) < -10 < 10 < centroid(w2t(Ew * (w > 0)))


def test_pulse_dispersion_reads_back_what_was_applied():
    t, w = grid(512)
    gdd, tod = pulse_dispersion(chirped(t, w, gdd=400.0, tod=2000.0), w)
    assert gdd == pytest.approx(400.0, rel=1e-3)
    assert tod == pytest.approx(2000.0, rel=1e-2)


def test_modes_of_the_applied_phase():
    _, w = grid()
    n_of = lambda wl: 1.45 + 3e3 / wl ** 2 + 1e9 / wl ** 4  # noqa: E731
    sp = stack_phase([(n_of, 2.0)], W800 + w * 0.2, W800)
    dw = sp.w_abs - W800
    assert sp.applied("gdd") == pytest.approx(sp.gdd / 2 * dw ** 2)
    full = sp.applied("full")
    assert full[w.size // 2] == pytest.approx(0, abs=1e-9)     # no constant…
    near = np.abs(dw) < 0.02
    assert np.allclose(full[near], sp.applied("gdd_tod")[near], atol=1e-4)  # …nor delay


def test_time_reversal_is_the_shg_ambiguity():
    t, w = grid(128, 2.0)
    E = chirped(t, w, fwhm=25.0, gdd=300.0, tod=3000.0)
    assert np.allclose(shg_trace(E), shg_trace(time_reverse(E)), atol=1e-12)
    # …and it negates the spectral phase
    assert pulse_dispersion(time_reverse(E), w)[0] == pytest.approx(-300.0, rel=1e-2)
    assert np.allclose(time_reverse(time_reverse(E)), E)


def test_trace_errors():
    T = np.random.default_rng(0).random((16, 16))
    assert trace_errors(3 * T, T) == pytest.approx((0.0, 0.0), abs=1e-12)
    G, Gp = trace_errors(T, T + 0.1)
    assert G > 0 and Gp > 0


@pytest.mark.parametrize("reverse", [False, True])
def test_direction_is_found(reverse):
    t, w = grid(128, 2.0)
    E_true = chirped(t, w, fwhm=25.0, gdd=-250.0)
    glass = 150.0 / 2 * w ** 2                     # 150 fs² of "glass"
    meas = shg_trace(propagate(E_true, glass))
    E_ret = time_reverse(E_true) if reverse else E_true
    G_as = trace_errors(meas, shg_trace(propagate(E_ret, glass)))[0]
    G_rev = trace_errors(meas, shg_trace(propagate(time_reverse(E_ret), glass)))[0]
    assert verdict(G_as, G_rev) == ("time-reversed" if reverse else "as retrieved")
    s = np.linspace(-2.5, 2.5, 51)
    assert best_s(s, scan(E_ret, glass, meas, s)) == pytest.approx(-1 if reverse else 1,
                                                                    abs=0.05)


def test_too_little_glass_is_inconclusive():
    assert verdict(1.00e-2, 1.05e-2) == "inconclusive"
    assert verdict(1.0e-2, 2.0e-2) == "as retrieved"
    assert verdict(2.0e-2, 1.0e-2) == "time-reversed"


def test_measured_trace_lands_on_the_pulse_grid():
    """End to end through the preparation pipeline: a .dwc measured after
    3 mm of fused silica, prepared onto the "before" pulse's grid."""
    from tadpole.prep.model import PrepSettings
    from tadpole.prep.pipeline import prepare, regrid
    from tadpole.prep.simulate import simulate_raw
    before, _ = prepare(simulate_raw(gdd_fs2=600.0, noise=0.0), PrepSettings(grid_n=128))
    g = (before.n, before.dt_fs, before.w0_rad_fs)
    el = M.Element("main", "SiO2", "Malitson", 3.0)
    w_abs = before.w0_rad_fs / 2 + before.w_rad_fs
    sp = stack_phase([(M.index_function(el), 3.0)], w_abs, before.w0_rad_fs / 2)
    # simulate_raw's gdd is the opposite sign of the material sense
    after_raw = simulate_raw(gdd_fs2=600.0 - sp.gdd, tod_fs3=-sp.tod, noise=0.0)
    after, _ = prepare(after_raw, PrepSettings(), grid=g)
    assert after.trace.shape == before.trace.shape
    assert after.dt_fs == before.dt_fs and after.w0_rad_fs == before.w0_rad_fs
    # the "retrieved" pulse: simulate_raw's field, on the grid
    t, w = before.t_fs, before.w_rad_fs
    E = chirped(t, w, fwhm=30.0, gdd=-600.0)
    G_as = trace_errors(after.trace, shg_trace(propagate(E, sp.applied("full"))),
                        after.mask)[0]
    G_rev = trace_errors(after.trace, shg_trace(propagate(time_reverse(E),
                                                          sp.applied("full"))),
                         after.mask)[0]
    assert verdict(G_as, G_rev) == "as retrieved"
    assert G_as < 5e-3
    # a prepared trace on another grid regrids onto this one
    other, _ = prepare(after_raw, PrepSettings(grid_n=256))
    rg = regrid(other, *g)
    assert rg.trace.shape == (128, 128) and rg.w0_rad_fs == before.w0_rad_fs
    assert trace_errors(after.trace, rg.trace, after.mask & rg.mask)[1] < 0.05
