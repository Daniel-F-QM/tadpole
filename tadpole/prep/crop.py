"""Cropping the raw trace to a wavelength and delay window."""

from __future__ import annotations

import numpy as np


def crop_window(axis: np.ndarray, lo: float | None, hi: float | None) -> slice:
    """The contiguous index range of `axis` (monotonic) inside [lo, hi].
    None on either side means unbounded. Never returns an empty slice: a
    window that misses the axis keeps the whole axis."""
    a = np.asarray(axis, float)
    keep = np.ones(a.size, bool)
    if lo is not None:
        keep &= a >= lo
    if hi is not None:
        keep &= a <= hi
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        return slice(0, a.size)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def crop(trace, wavelengths, delays, wl_min=None, wl_max=None,
         d_min=None, d_max=None):
    """(trace, wavelengths, delays) restricted to the window."""
    sw = crop_window(wavelengths, wl_min, wl_max)
    sd = crop_window(delays, d_min, d_max)
    return trace[sw, sd], np.asarray(wavelengths)[sw], np.asarray(delays)[sd]
