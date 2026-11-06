"""Putting a measured trace onto an N×N Fourier grid.

The spectrometer samples uniformly in WAVELENGTH, the algorithms need uniform
ANGULAR FREQUENCY with

    dω = 2π / (N·dt),     delays τ_k = (k − N/2)·dt

i.e. the delay step equals the time step and Δt·Δν = 1/N. Converting a
spectral density from λ to ω needs the Jacobian |dλ/dω| = λ²/(2πc), or the
spectrum's long-wavelength side comes out too strong.
"""

from __future__ import annotations

import numpy as np

from tadpole.prep.model import C_NM_PER_FS

TWO_PI_C = 2 * np.pi * C_NM_PER_FS     # ω [rad/fs] = TWO_PI_C / λ [nm]


def wavelength_to_frequency(trace: np.ndarray, wl_nm: np.ndarray):
    """(trace_w, w) with w = 2πc/λ ASCENDING and the λ² Jacobian applied, so
    ∫ I dω = ∫ I dλ. Rows of `trace` follow `wl_nm`."""
    wl = np.asarray(wl_nm, float)
    w = TWO_PI_C / wl
    T = np.asarray(trace, float) * (wl ** 2 / TWO_PI_C)[:, None]
    order = np.argsort(w)
    return T[order], w[order]


def centroid(x: np.ndarray, y: np.ndarray) -> float:
    y = np.clip(np.asarray(y, float), 0, None)
    s = y.sum()
    return float((x * y).sum() / s) if s > 0 else float(np.mean(x))


def signal_span(x: np.ndarray, y: np.ndarray, frac: float = 0.02) -> float:
    """Width of the region of `x` where the marginal `y` exceeds `frac` of its
    peak — how much of the axis actually carries signal."""
    y = np.asarray(y, float)
    m = y.max() if y.size else 0.0
    if m <= 0:
        return float(np.ptp(x))
    idx = np.flatnonzero(y > frac * m)
    return float(abs(x[idx[-1]] - x[idx[0]])) if idx.size > 1 else float(abs(x[1] - x[0]))


def auto_dt(delay_span_fs: float, freq_span_rad_fs: float, n: int,
            margin: float = 1.5) -> float:
    """The time step that covers BOTH spans with the same relative margin.

    The delay window is N·dt and the frequency window 2π/dt, so covering
    D·margin in delay needs dt ≥ D·margin/N and covering S·margin in frequency
    needs dt ≤ 2π/(S·margin). The geometric mean of the two bounds leaves the
    same headroom on both axes; when the bounds cross, N is too small for this
    trace and no dt fits both (prep reports the shortfall).
    """
    lo = delay_span_fs * margin / n
    hi = 2 * np.pi / (max(freq_span_rad_fs, 1e-12) * margin)
    return float(np.sqrt(max(lo, 1e-12) * hi))


def resample_axis(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray,
                  axis: int = 0):
    """Resample `y_src` (sampled at monotonic ascending `x_src` along `axis`)
    onto `x_dst`. Returns (values, inside) where `inside` marks destination
    points within the source range; outside points are 0.

    When the destination is COARSER than the source (step > 2× the median
    source step) every destination value is the average over its bin, via the
    cumulative integral — linear interpolation would pick one noisy sample out
    of many. Otherwise linear interpolation.
    """
    xs = np.asarray(x_src, float)
    xd = np.asarray(x_dst, float)
    y = np.moveaxis(np.asarray(y_src, float), axis, 0)
    flat = y.reshape(y.shape[0], -1)
    inside = (xd >= xs[0]) & (xd <= xs[-1])
    out = np.zeros((xd.size, flat.shape[1]))
    step_src = np.median(np.diff(xs)) if xs.size > 1 else 1.0
    step_dst = np.median(np.abs(np.diff(xd))) if xd.size > 1 else step_src
    if step_dst > 2 * step_src:
        # cumulative trapezoid integral per column
        dx = np.diff(xs)[:, None]
        F = np.vstack([np.zeros((1, flat.shape[1])),
                       np.cumsum(0.5 * (flat[1:] + flat[:-1]) * dx, axis=0)])
        a = np.clip(xd - step_dst / 2, xs[0], xs[-1])
        b = np.clip(xd + step_dst / 2, xs[0], xs[-1])
        for c in range(flat.shape[1]):
            Fa = np.interp(a, xs, F[:, c])
            Fb = np.interp(b, xs, F[:, c])
            width = np.where(b > a, b - a, 1.0)
            out[:, c] = np.where(b > a, (Fb - Fa) / width, 0.0)
    else:
        for c in range(flat.shape[1]):
            out[:, c] = np.interp(xd, xs, flat[:, c], left=0.0, right=0.0)
    out[~inside] = 0.0
    out = out.reshape((xd.size,) + y.shape[1:])
    return np.moveaxis(out, 0, axis), inside


def symmetrize_delay(trace: np.ndarray) -> np.ndarray:
    """Average each column τ with its mirror −τ (column k ↔ N − k; column 0,
    the most negative delay, has no partner and is kept)."""
    T = np.array(trace, float, copy=True)
    n = T.shape[1]
    k = np.arange(1, n)
    T[:, k] = 0.5 * (trace[:, k] + trace[:, n - k])
    return T
