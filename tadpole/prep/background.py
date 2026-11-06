"""Background removal on the raw trace (rows wavelength, columns delay).

Far from zero delay an SHG trace carries no signal, so the edge delay columns
are a direct measurement of the background: detector dark level, stray light,
and the unmodulated second harmonic of each arm on its own.
"""

from __future__ import annotations

import numpy as np

BG_MODES = ("none", "constant", "spectral")


def subtract_background(trace: np.ndarray, mode: str = "spectral",
                        edge_cols: int = 5, edge_rows: int = 0,
                        clip_negative: bool = True) -> np.ndarray:
    """Return a background-subtracted copy of `trace`.

    * "constant": one value, the mean of the `edge_cols` columns at each end
    * "spectral": per wavelength row, the mean of those edge columns — the
      background spectrum
    * `edge_rows` > 0: afterwards, per delay column, subtract the mean of that
      many rows at each end of the wavelength axis (removes a delay-dependent
      offset such as a drifting dark level)
    """
    if mode not in BG_MODES:
        raise ValueError(f"unknown background mode {mode!r}")
    T = np.array(trace, dtype=float, copy=True)
    n_wl, n_d = T.shape
    k = int(max(0, min(edge_cols, n_d // 2)))
    if mode != "none" and k > 0:
        edges = np.concatenate([T[:, :k], T[:, n_d - k:]], axis=1)
        if mode == "constant":
            T -= edges.mean()
        else:
            T -= edges.mean(axis=1, keepdims=True)
    r = int(max(0, min(edge_rows, n_wl // 2)))
    if r > 0:
        rows = np.concatenate([T[:r, :], T[n_wl - r:, :]], axis=0)
        T -= rows.mean(axis=0, keepdims=True)
    if clip_negative:
        np.maximum(T, 0.0, out=T)
    return T
