"""A synthetic MEASURED trace (wavelength × delay, with background and noise),
for trying the whole pipeline without a .dwc file. numpy only.

It mimics what Lillypad records: an SHG FROG trace sampled uniformly in
wavelength by a spectrometer, a constant dark level, a weak background
spectrum, and shot-like noise.
"""

from __future__ import annotations

import numpy as np

from tadpole.io.dwc import RawTrace

C = 299.792458   # nm/fs


def _t2w(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def _w2t(x):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))


def simulate_raw(center_nm=800.0, fwhm_fs=30.0, gdd_fs2=600.0, tod_fs3=0.0,
                 delay_step_fs=2.0, n_delays=151, wl_range_nm=(360.0, 440.0),
                 n_wl=1024, dark=50.0, peak_counts=4000.0, noise=0.01,
                 seed=0) -> RawTrace:
    rng = np.random.default_rng(seed)
    n, dt = 4096, 0.25
    t = (np.arange(n) - n // 2) * dt
    w = (np.arange(n) - n // 2) * 2 * np.pi / (n * dt)
    E = np.exp(-2 * np.log(2) * (t / fwhm_fs) ** 2).astype(complex)
    Ew = _t2w(E) * np.exp(1j * (gdd_fs2 / 2 * w ** 2 + tod_fs3 / 6 * w ** 3))
    E = _w2t(Ew)

    w0 = 2 * np.pi * C / center_nm
    wl = np.linspace(*wl_range_nm, n_wl)
    w_sig = 2 * np.pi * C / wl - 2 * w0          # signal offsets at each pixel
    step = int(round(delay_step_fs / dt))
    shifts = (np.arange(n_delays) - n_delays // 2) * step
    T = np.empty((n_wl, n_delays))
    for j, s in enumerate(shifts):
        Iw = np.abs(_t2w(E * np.roll(E, s))) ** 2
        # density per unit λ: I_λ = I_ω · |dω/dλ| = I_ω · 2πc/λ²
        T[:, j] = np.interp(w_sig, w, Iw) * (2 * np.pi * C / wl ** 2)
    T *= peak_counts / T.max()
    bg_spectrum = 0.02 * peak_counts * np.exp(-((wl - wl.mean()) / (0.3 * np.ptp(wl))) ** 2)
    T += dark + bg_spectrum[:, None]
    T += noise * peak_counts * rng.standard_normal(T.shape)
    delays = (np.arange(n_delays) - n_delays // 2) * delay_step_fs
    return RawTrace(trace=T, wavelengths_nm=wl, delays_fs=delays,
                    source="simulated",
                    meta={"fwhm_fs": fwhm_fs, "gdd_fs2": gdd_fs2, "tod_fs3": tod_fs3,
                          "center_nm": center_nm})
