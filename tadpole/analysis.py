"""Display-side physics on a retrieved field: spectra, FWHM, blanked phases.

No Qt. Uses the same transform convention as the Julia package
(t → ω with kernel e^{-iωt}: fftshift(fft(ifftshift(x)))).
"""

from __future__ import annotations

import numpy as np

C_NM_PER_FS = 299.792458


def t2w(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Time → frequency along `axis` (kernel e^{-iωt}), centred both sides."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis),
                           axes=axis)


def w2t(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """The inverse of `t2w`."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis),
                           axes=axis)


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """FWHM of the main peak by linear interpolation of the half-max points;
    nan if the peak touches the edge."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.size < 3 or not np.any(y > 0):
        return float("nan")
    i = int(np.argmax(y)); h = y[i] / 2
    l = i
    while l > 0 and y[l] > h:
        l -= 1
    r = i
    while r < y.size - 1 and y[r] > h:
        r += 1
    if y[l] > h or y[r] > h:
        return float("nan")
    xl = x[l] + (h - y[l]) / (y[l + 1] - y[l]) * (x[l + 1] - x[l])
    xr = x[r - 1] + (y[r - 1] - h) / (y[r - 1] - y[r]) * (x[r] - x[r - 1])
    return float(abs(xr - xl))


def blanked_phase(field: np.ndarray, rel_threshold: float = 0.01) -> np.ndarray:
    """Unwrapped phase, NaN where |field|² < threshold·peak, zero at the peak."""
    I = np.abs(field) ** 2
    ph = np.unwrap(np.angle(field))
    ph = ph - ph[int(np.argmax(I))]
    ph = ph.astype(float)
    ph[I < rel_threshold * I.max()] = np.nan
    return ph


def visible_range(x: np.ndarray, y: np.ndarray, rel: float = 1e-3, pad: float = 0.15):
    """(lo, hi) of x where y exceeds rel·max, widened by `pad` of the span."""
    idx = np.flatnonzero(np.asarray(y) > rel * np.max(y))
    if idx.size < 2:
        return float(x[0]), float(x[-1])
    lo, hi = float(x[idx[0]]), float(x[idx[-1]])
    span = hi - lo
    return lo - pad * span, hi + pad * span


def centred_range(x: np.ndarray, y: np.ndarray, centre: float,
                  rel: float = 1e-3, pad: float = 0.1):
    """`visible_range`, made symmetric about `centre`: wide enough for the
    signal on both sides, so the trace sits in the middle of the view. (Noise
    above `rel` on one side only would otherwise push it off-centre.) Never
    wider than the axis `x` itself."""
    lo, hi = visible_range(x, y, rel, pad)
    half = max(centre - lo, hi - centre)
    half = min(half, max(centre - float(np.min(x)), float(np.max(x)) - centre))
    return centre - half, centre + half


def wavelength_axis(w: np.ndarray, w0_signal: float):
    """(λ ascending in nm, mask of grid points with ω > 0, sort order) for the
    PULSE spectrum on offset grid `w` (carrier w0_signal/2)."""
    w_abs = w0_signal / 2 + np.asarray(w, float)
    pos = w_abs > 0
    lam = 2 * np.pi * C_NM_PER_FS / w_abs[pos]
    order = np.argsort(lam)
    return lam[order], pos, order


def pulse_views(E: np.ndarray, t: np.ndarray, w: np.ndarray, w0_signal: float) -> dict:
    """Everything the temporal/spectral panels show, from E(t).

    The SHG signal is centred at w0_signal, so the pulse's carrier is half of
    that; the pulse spectrum lives on the same offset grid `w`.
    """
    E = np.asarray(E)
    It = np.abs(E) ** 2
    peak = It.max()
    It = It / peak
    Ew = t2w(E)
    Iw = np.abs(Ew) ** 2
    Iw = Iw / Iw.max()
    phi_w = blanked_phase(Ew)
    w_abs = w0_signal / 2 + np.asarray(w)
    # The same spectrum against wavelength, as a spectrometer shows it: a
    # density per nm (Jacobian |dω/dλ| = 2πc/λ² ∝ ω²), ascending λ, only
    # where ω > 0 has a wavelength.
    lam, pos, order = wavelength_axis(w, w0_signal)
    Il = (Iw * w_abs ** 2)[pos][order]
    Il = Il / Il.max() if Il.size and Il.max() > 0 else Il
    # Transform limit: the same spectrum with a flat phase. Same energy
    # (Parseval), so its peak relative to the pulse's is the peak-power gain.
    It_tl = np.abs(w2t(np.abs(Ew))) ** 2 / peak
    return {
        "t": np.asarray(t), "It": It, "phi_t": blanked_phase(E),
        "w": w_abs, "Iw": Iw, "phi_w": phi_w,
        "lam": lam, "Il": Il, "phi_lam": phi_w[pos][order],
        "fwhm_t": fwhm(t, It), "fwhm_w": fwhm(w_abs, Iw), "fwhm_lam": fwhm(lam, Il),
        "It_tl": It_tl, "fwhm_tl": fwhm(t, It_tl), "tl_peak": float(It_tl.max()),
        "lambda0_nm": 2 * np.pi * C_NM_PER_FS / (w0_signal / 2),
    }
