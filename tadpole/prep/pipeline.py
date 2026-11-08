"""The full preparation pipeline: raw .dwc trace → PreparedTrace.

    crop → background → λ→ω (+ Jacobian) → delay zero → centre frequency
    → time step → resample onto the N×N grid → symmetrize → normalise

Pure numpy: the Prepare page calls `prepare` on every settings change, and the
tests call it with synthetic traces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tadpole.io.dwc import RawTrace
from tadpole.prep.background import subtract_background
from tadpole.prep.crop import crop
from tadpole.prep.grid import (auto_dt, centroid, resample_axis, signal_span,
                               symmetrize_delay, wavelength_to_frequency)
from tadpole.prep.model import PreparedTrace, PrepSettings

# Above this delay asymmetry Prepare warns. A clean simulated trace with 1%
# noise measures ~0.015.
ASYMMETRY_WARN = 0.15


@dataclass
class PrepReport:
    """What the Prepare page shows beside the result, and warns about."""
    delay_span_fs: float          # where the delay marginal carries signal
    freq_span_rad_fs: float       # where the spectral marginal carries signal
    delay_window_fs: float        # N·dt
    freq_window_rad_fs: float     # 2π/dt
    zero_shift_fs: float          # how far the delay axis was moved
    energy_inside: float          # fraction of the trace's energy on the grid
    asymmetry: float              # ‖T(τ) − T(−τ)‖/‖T‖ on the grid, before
                                  # symmetrizing; an ideal SHG trace gives ~0
    warnings: list[str]


def background_only(raw: RawTrace, s: PrepSettings):
    """(trace, wavelengths, delays) after crop + background: the left-hand
    "processed" preview, still on the measured axes."""
    T, wl, d = crop(raw.trace, raw.wavelengths_nm, raw.delays_fs,
                    s.wl_min_nm, s.wl_max_nm, s.delay_min_fs, s.delay_max_fs)
    # Negative pixels are KEPT here. Noise around zero averages out when the
    # trace is binned onto the grid; clipping it first turns the noise into a
    # positive floor under the whole trace, which biases the retrieved pulse
    # (≈10% too short on a 1%-noise test trace) and the automatic dt. The
    # trace is clipped once, on the grid, below.
    T = subtract_background(T, s.bg_mode, s.bg_edge_cols, s.bg_edge_rows,
                            clip_negative=False)
    return T, wl, d


def find_zero(delays: np.ndarray, marginal: np.ndarray, mode: str) -> float:
    if mode == "centroid":
        return centroid(delays, marginal)
    if mode == "peak":
        return float(delays[int(np.argmax(marginal))])
    if mode == "none":
        return 0.0
    raise ValueError(f"unknown zero mode {mode!r}")


def prepare(raw: RawTrace, s: PrepSettings, grid=None) -> tuple[PreparedTrace, PrepReport]:
    """`grid=(n, dt_fs, w0_rad_fs)` forces the Fourier grid instead of choosing
    it — to put a second measurement onto an existing pulse's grid (the
    Propagator's comparison trace)."""
    T, wl, d = background_only(raw, s)
    Tw, w = wavelength_to_frequency(T, wl)

    # Delay zero from the delay marginal (∝ autocorrelation for SHG).
    shift = find_zero(d, Tw.sum(axis=0), s.zero_mode) + s.zero_offset_fs
    d = d - shift
    if d.size > 1 and d[-1] < d[0]:
        d, Tw = d[::-1], Tw[:, ::-1]

    # Centre frequency from the spectral marginal.
    spec = Tw.sum(axis=1)
    w0 = centroid(w, spec) if grid is None else float(grid[2])

    n = int(s.grid_n) if grid is None else int(grid[0])
    if n < 8 or n % 2:
        raise ValueError(f"grid size must be even and ≥ 8, got {n}")
    d_span = signal_span(d, Tw.sum(axis=0))
    w_span = signal_span(w, spec)
    if grid is not None:
        dt = float(grid[1])
    else:
        dt = float(s.grid_dt_fs) if s.grid_dt_fs > 0 else auto_dt(d_span, w_span, n)
    dw = 2 * np.pi / (n * dt)
    k = np.arange(n) - n // 2
    tau, wg = k * dt, k * dw

    # frequency first (onto w0 + wg), then delay
    Tf, in_w = resample_axis(w, Tw, w0 + wg, axis=0)
    Tg, in_d = resample_axis(d, Tf, tau, axis=1)
    mask = in_w[:, None] & in_d[None, :]
    nrm = np.linalg.norm(Tg)
    asymmetry = float(2 * np.linalg.norm(Tg - symmetrize_delay(Tg)) / nrm) if nrm > 0 else 0.0
    if s.symmetrize:
        Tg = symmetrize_delay(Tg)
        mask = mask | symmetrize_delay(mask.astype(float)).astype(bool)
    Tg = np.clip(Tg, 0, None)        # after binning, see background_only
    peak = Tg.max()
    if peak <= 0:
        raise ValueError("the prepared trace is empty — check the crop window "
                         "and background settings")
    Tg /= peak
    Tf_meas = np.clip(Tf, 0, None) / peak
    mask_meas = np.repeat(in_w[:, None], d.size, axis=1)

    # Energy bookkeeping: how much of the measured signal the grid holds.
    total = np.trapezoid(np.trapezoid(Tw, w, axis=0), d) if d.size > 1 else 1.0
    on_grid = Tg.sum() * peak * dw * dt
    energy_inside = float(on_grid / total) if total > 0 else 0.0

    warnings = []
    if n * dt < 1.2 * d_span:
        warnings.append(f"Delay window N·dt = {n*dt:.0f} fs is barely wider than "
                        f"the signal ({d_span:.0f} fs): increase N or dt.")
    if 2 * np.pi / dt < 1.2 * w_span:
        warnings.append(f"Frequency window 2π/dt = {2*np.pi/dt:.3f} rad/fs is "
                        f"barely wider than the spectrum ({w_span:.3f} rad/fs): "
                        f"increase N or reduce dt.")
    if energy_inside < 0.95:
        warnings.append(f"Only {100*energy_inside:.0f}% of the trace's energy "
                        f"lands on the grid.")
    if asymmetry > ASYMMETRY_WARN:
        warnings.append(f"The trace is {100*asymmetry:.0f}% asymmetric in delay; an "
                        f"SHG trace should be symmetric. No SHG retrieval can fit "
                        f"that part (check alignment / delay zero), so expect a "
                        f"high G′.")
    if np.median(np.abs(np.diff(raw.delays_fs))) > 1.01 * dt:
        warnings.append("The grid's dt is finer than the measured delay step; "
                        "intermediate delays are interpolated.")

    prepared = PreparedTrace(
        trace=Tg, mask=mask, dt_fs=dt, w0_rad_fs=w0,
        trace_meas=Tf_meas, mask_meas=mask_meas, delays_meas_fs=d,
        source=raw.source, settings=s.to_dict())
    report = PrepReport(d_span, w_span, n * dt, 2 * np.pi / dt, shift,
                        energy_inside, asymmetry, warnings)
    return prepared, report


def regrid(p: PreparedTrace, n: int, dt_fs: float, w0_rad_fs: float) -> PreparedTrace:
    """A prepared trace resampled onto another N×N grid (frequency, then
    delay), peak 1 again. Pixels outside its own grid are masked out."""
    k = np.arange(n) - n // 2
    tau, wg = k * dt_fs, k * 2 * np.pi / (n * dt_fs)
    Tf, in_w = resample_axis(p.w0_rad_fs + p.w_rad_fs, p.trace, w0_rad_fs + wg, axis=0)
    Tg, in_d = resample_axis(p.t_fs, Tf, tau, axis=1)
    Mf, _ = resample_axis(p.w0_rad_fs + p.w_rad_fs, p.mask.astype(float),
                          w0_rad_fs + wg, axis=0)
    Mg, _ = resample_axis(p.t_fs, Mf, tau, axis=1)
    mask = (Mg > 0.5) & in_w[:, None] & in_d[None, :]
    Tg = np.clip(Tg, 0, None)
    peak = Tg.max()
    if peak <= 0:
        raise ValueError("the trace does not overlap the pulse's grid")
    return PreparedTrace(
        trace=Tg / peak, mask=mask, dt_fs=float(dt_fs), w0_rad_fs=float(w0_rad_fs),
        trace_meas=p.trace_meas, mask_meas=p.mask_meas,
        delays_meas_fs=p.delays_meas_fs, source=p.source, settings=p.settings)
