import numpy as np
import pytest

from tadpole.prep.background import subtract_background
from tadpole.prep.crop import crop_window
from tadpole.prep.grid import (auto_dt, resample_axis, symmetrize_delay,
                               wavelength_to_frequency)
from tadpole.prep.model import PrepSettings
from tadpole.prep.pipeline import prepare
from tadpole.prep.simulate import simulate_raw


def test_jacobian_conserves_energy():
    wl = np.linspace(350, 450, 2001)
    I = np.exp(-((wl - 400) / 10) ** 2)[:, None]
    Iw, w = wavelength_to_frequency(I, wl)
    assert np.all(np.diff(w) > 0)
    e_wl = np.trapezoid(I[:, 0], wl)
    e_w = np.trapezoid(Iw[:, 0], w)
    assert e_w == pytest.approx(e_wl, rel=1e-4)


def test_auto_dt_balances_both_windows():
    D, S, n = 200.0, 0.5, 128
    dt = auto_dt(D, S, n, margin=1.5)
    # same relative headroom on both axes
    assert (n * dt) / (1.5 * D) == pytest.approx((2 * np.pi / dt) / (1.5 * S))


def test_resample_bin_average_and_linear():
    x = np.linspace(0, 10, 1001)
    y = np.sin(x)[:, None]
    coarse = np.linspace(1, 9, 9)
    out, inside = resample_axis(x, y, coarse)
    assert inside.all()
    # the average of sin over a bin of width 1 is sin(x)·sin(½)/½
    np.testing.assert_allclose(out[:, 0], np.sin(coarse) * np.sin(0.5) / 0.5, atol=2e-3)
    fine = np.linspace(-1, 11, 5001)
    out, inside = resample_axis(x, y, fine)
    assert not inside[0] and out[0, 0] == 0.0
    np.testing.assert_allclose(out[inside, 0], np.sin(fine[inside]), atol=1e-4)


def test_symmetrize():
    T = np.arange(16.0).reshape(2, 8)
    S = symmetrize_delay(T)
    np.testing.assert_allclose(S[:, 1:], S[:, 1:][:, ::-1])
    np.testing.assert_allclose(S[:, 0], T[:, 0])


def test_background_modes():
    T = np.full((10, 20), 5.0)
    T[:, 8:12] += 100
    assert subtract_background(T, "constant", 3)[0, 0] == pytest.approx(0)
    spec = np.linspace(0, 9, 10)[:, None]
    out = subtract_background(T + spec, "spectral", 3)
    np.testing.assert_allclose(out[:, 0], 0, atol=1e-12)
    assert out[0, 10] == pytest.approx(100)
    assert subtract_background(T - 10, "none", clip_negative=True).min() == 0


def test_crop_window_never_empty():
    a = np.arange(10.0)
    assert crop_window(a, 2, 5) == slice(2, 6)
    assert crop_window(a, 50, 60) == slice(0, 10)
    assert crop_window(a, None, None) == slice(0, 10)


@pytest.mark.parametrize("n", [64, 128, 256])
def test_prepare_simulated(n):
    raw = simulate_raw(delay_step_fs=3.0, n_delays=101, seed=1)
    raw.delays_fs = raw.delays_fs + 12.0            # pretend the file is off-centre
    p, r = prepare(raw, PrepSettings(grid_n=n))
    assert p.trace.shape == (n, n)
    assert p.trace.max() == pytest.approx(1.0)
    assert p.dt_fs * p.dw_rad_fs / (2 * np.pi) == pytest.approx(1 / n)   # Δt·Δν = 1/N
    assert p.lambda0_nm == pytest.approx(400.0, abs=0.5)                 # SHG of 800 nm
    assert r.zero_shift_fs == pytest.approx(12.0, abs=0.5)
    assert r.energy_inside > 0.95
    assert r.asymmetry < 0.05 and not any("asymmetric" in w for w in r.warnings)
    assert p.trace_meas.shape == (n, 101) and p.delays_meas_fs.size == 101


def test_prepare_rejects_bad_grid():
    with pytest.raises(ValueError):
        prepare(simulate_raw(), PrepSettings(grid_n=7))


def test_asymmetric_trace_warns():
    raw = simulate_raw(noise=0.0)
    raw.trace[:, : raw.trace.shape[1] // 2] *= 0.5      # weaker on one side
    _, r = prepare(raw, PrepSettings())
    assert r.asymmetry > 0.15
    assert any("asymmetric" in w for w in r.warnings)
