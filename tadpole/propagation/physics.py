"""Material dispersion, propagation and the SHG forward model. numpy only.

Conventions (shared with the Julia package, see ARCHITECTURE.md):

* `t2w` uses the kernel e^{-iωt}, so E(t) = ∫ Ẽ(ω) e^{+iωt} dω.
* Material adds k(ω)·L. With this kernel a field after the material is
  Ẽ(ω)·e^{−i·φ(ω)}, φ = n(ω)·ω·L/c, so that positive GDD delays blue behind
  red (normal dispersion). A test pins this sign.
* "GDD" everywhere — material or pulse — is d²φ/dω² in that same sense:
  positive is "like glass". A pulse's GDD is −d²ψ/dω² of the phase ψ of its
  own spectrum t2w(E), so pulse GDD + stack GDD is the GDD after the stack.
* SHG trace: S[i, k] = E[i]·E[(i − m_k) mod N], m_k = k − N/2, then t2w down
  the rows; rows are frequency, columns delay — Julia's `frog_trace`.

The time-reversal ambiguity: E*(−t) has the same SHG trace as E(t) and the
negated spectral phase. Hence (reversed + stack) has the trace of
(as-retrieved − stack), and one scan of the error against a signed multiple s
of the stack's phase tells the direction: a minimum near s = +1 means the
retrieved direction is right, near s = −1 that it is time-reversed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tadpole.analysis import t2w, w2t

C_NM_PER_FS = 299.792458
TWO_PI_C = 2 * np.pi * C_NM_PER_FS

STENCIL_H = 0.01           # rad/fs, finite-difference step for the Taylor terms
MODES = {"full": "Full n(ω)", "gdd": "GDD only", "gdd_tod": "GDD + TOD"}


# ── material phase ──────────────────────────────────────────────────────────
def material_phase(n_of, thickness_mm: float, w_abs) -> np.ndarray:
    """φ(ω) = n(ω)·ω·L/c in rad. `n_of` maps λ [nm] → n (NaN where unknown);
    ω ≤ 0 has no wavelength and comes out NaN."""
    w = np.asarray(w_abs, float)
    pos = w > 0
    wl = np.where(pos, TWO_PI_C / np.where(pos, w, 1.0), np.nan)
    n = np.asarray(n_of(wl), float)
    return n * w * (thickness_mm * 1e6) / C_NM_PER_FS


def taylor(phase_fn, w0: float, h: float = STENCIL_H) -> np.ndarray:
    """(φ, φ′, φ″, φ‴) at ω₀ from a 5-point stencil: phase, group delay (fs),
    GDD (fs²), TOD (fs³)."""
    f = np.asarray(phase_fn(w0 + h * np.arange(-2, 3)), float)
    if not np.all(np.isfinite(f)):
        raise ValueError("no refractive-index data at the pulse's centre wavelength "
                         f"({TWO_PI_C / w0:.0f} nm)")
    return np.array([
        f[2],
        (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h),
        (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h ** 2),
        (-f[0] + 2 * f[1] - 2 * f[3] + f[4]) / (2 * h ** 3),
    ])


def _poly(c, dw):
    return c[0] + c[1] * dw + c[2] / 2 * dw ** 2 + c[3] / 6 * dw ** 3


@dataclass
class ElementPhase:
    coeffs: np.ndarray            # (φ, GD, GDD, TOD) at ω₀
    filled: np.ndarray            # bool per frequency: no data, Taylor-filled


@dataclass
class StackPhase:
    """The whole stack's phase on a pulse grid."""
    w_abs: np.ndarray
    w0: float
    total: np.ndarray             # Σ φ_j(ω), NaN-free
    coeffs: np.ndarray            # Σ of the elements' Taylor terms
    elements: list[ElementPhase] = field(default_factory=list)

    @property
    def gdd(self) -> float:
        return float(self.coeffs[2])

    @property
    def tod(self) -> float:
        return float(self.coeffs[3])

    def applied(self, mode: str = "full") -> np.ndarray:
        """The phase actually applied: always without the constant and the
        group delay at ω₀, so the pulse stays centred on the grid."""
        dw = self.w_abs - self.w0
        c = self.coeffs
        if mode == "full":
            return self.total - c[0] - c[1] * dw
        if mode == "gdd":
            return c[2] / 2 * dw ** 2
        if mode == "gdd_tod":
            return c[2] / 2 * dw ** 2 + c[3] / 6 * dw ** 3
        raise ValueError(f"unknown dispersion mode {mode!r}")


def stack_phase(items, w_abs, w0: float) -> StackPhase:
    """`items`: (n_of, thickness_mm) per element. Frequencies without data
    (outside a page's range) are filled from that element's cubic Taylor
    expansion at ω₀ and flagged."""
    w = np.asarray(w_abs, float)
    total = np.zeros_like(w)
    coeffs = np.zeros(4)
    parts = []
    for n_of, L in items:
        fn = lambda x, n_of=n_of, L=L: material_phase(n_of, L, x)  # noqa: E731
        c = taylor(fn, w0)
        phi = fn(w)
        bad = ~np.isfinite(phi)
        phi[bad] = _poly(c, w[bad] - w0)
        total += phi
        coeffs += c
        parts.append(ElementPhase(c, bad))
    return StackPhase(w, float(w0), total, coeffs, parts)


# ── fields and traces ───────────────────────────────────────────────────────
def propagate(E, phase) -> np.ndarray:
    """E(t) after a material of spectral phase φ(ω) (on the pulse grid)."""
    return w2t(t2w(np.asarray(E, complex)) * np.exp(-1j * np.asarray(phase)))


def time_reverse(E) -> np.ndarray:
    """E*(−t) on the grid: index i ↔ N − i; index 0 (t = −N/2·dt) has no
    partner and is kept, as in Julia's `pulse_distance`."""
    E = np.asarray(E, complex)
    out = np.empty_like(E)
    out[0] = np.conj(E[0])
    out[1:] = np.conj(E[:0:-1])
    return out


def shg_trace(E) -> np.ndarray:
    """|t2w(E(t)·E(t − τ))|², rows ω and columns τ, peak 1 (Julia's
    `frog_trace(SHG(), E)`, normalised)."""
    E = np.asarray(E, complex)
    n = E.size
    m = np.arange(n) - n // 2
    idx = (np.arange(n)[:, None] - m[None, :]) % n
    T = np.abs(t2w(E[:, None] * E[idx], axis=0)) ** 2
    peak = T.max()
    return T / peak if peak > 0 else T


def trace_errors(meas, model, mask=None) -> tuple[float, float]:
    """(G, G′) with the least-squares scale μ — the definitions in Julia's
    common.jl: G = √(Σ(I−μT)²/n), G′ = √(Σ(I−μT)²/ΣI²)."""
    I = np.asarray(meas, float)
    T = np.asarray(model, float)
    if mask is not None:
        I, T = I[mask], T[mask]
    I, T = I.ravel(), T.ravel()
    tt = float(T @ T)
    mu = float(I @ T) / tt if tt > 0 else 0.0
    s = float(np.sum((I - mu * T) ** 2))
    return float(np.sqrt(s / max(I.size, 1))), float(np.sqrt(s / max(float(I @ I), 1e-300)))


def pulse_dispersion(E, w_off) -> tuple[float, float]:
    """The pulse's own (GDD, TOD) in the material sense (see module doc):
    a cubic fit to its spectral phase, weighted by the spectrum, over where
    the spectrum exceeds 5 % of its peak."""
    Ew = t2w(np.asarray(E, complex))
    I = np.abs(Ew) ** 2
    sel = I > 0.05 * I.max()
    if sel.sum() < 5:
        return float("nan"), float("nan")
    w = np.asarray(w_off, float)[sel]
    psi = np.unwrap(np.angle(Ew[sel]))
    c3, c2, _, _ = np.polyfit(w, psi, 3, w=np.sqrt(I[sel]))
    return float(-2 * c2), float(-6 * c3)


# ── comparison with a measurement ───────────────────────────────────────────
def scan(E, phase, meas, s_values, mask=None) -> np.ndarray:
    """G of the as-retrieved pulse after s·phase, for each s."""
    return np.array([trace_errors(meas, shg_trace(propagate(E, s * phase)), mask)[0]
                     for s in s_values])


def best_s(s_values, G) -> float:
    """Minimum of G(s), refined by a parabola through its neighbours."""
    s = np.asarray(s_values, float)
    G = np.asarray(G, float)
    i = int(np.argmin(G))
    if 0 < i < s.size - 1:
        d = G[i - 1] - 2 * G[i] + G[i + 1]
        if d > 0:
            return float(s[i] + 0.5 * (s[i + 1] - s[i]) * (G[i - 1] - G[i + 1]) / d)
    return float(s[i])


INCONCLUSIVE_REL = 0.10


def verdict(G_as: float, G_rev: float, rel: float = INCONCLUSIVE_REL) -> str:
    """"as retrieved", "time-reversed", or "inconclusive" when the two
    errors differ by less than `rel` (e.g. too little material to tell)."""
    lo = min(G_as, G_rev)
    if not np.isfinite(lo) or abs(G_as - G_rev) <= rel * max(lo, 1e-300):
        return "inconclusive"
    return "as retrieved" if G_as < G_rev else "time-reversed"
