"""Saving and loading prepared traces and retrieval results (.npz / .csv)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tadpole.prep.model import PreparedTrace


def save_prepared(path, p: PreparedTrace) -> None:
    np.savez_compressed(
        path, kind="tadpole_prepared", trace=p.trace, mask=p.mask,
        dt_fs=p.dt_fs, w0_rad_fs=p.w0_rad_fs, trace_meas=p.trace_meas,
        mask_meas=p.mask_meas, delays_meas_fs=p.delays_meas_fs,
        source=p.source, settings=json.dumps(p.settings))


def load_prepared(path) -> PreparedTrace:
    with np.load(path, allow_pickle=False) as z:
        if str(z.get("kind", "")) != "tadpole_prepared":
            raise ValueError(f"{Path(path).name} is not a Tadpole prepared trace")
        return PreparedTrace(
            trace=z["trace"], mask=z["mask"].astype(bool), dt_fs=float(z["dt_fs"]),
            w0_rad_fs=float(z["w0_rad_fs"]), trace_meas=z["trace_meas"],
            mask_meas=z["mask_meas"].astype(bool), delays_meas_fs=z["delays_meas_fs"],
            source=str(z["source"]), settings=json.loads(str(z["settings"])))


def save_result(path, prepared: PreparedTrace, result: dict) -> None:
    """A retrieval result: .npz (everything) or .csv (the field only).

    `result` is the engine's Done payload: E, trace, errors, errors_p, G, Gp,
    t, w, algorithm, iterations, stopped.
    """
    path = Path(path)
    E = np.asarray(result["E"])
    t = np.asarray(result["t"])
    if path.suffix.lower() == ".csv":
        Ew = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E)))
        w = np.asarray(result["w"])
        # The pulse's carrier is half the SHG signal's.
        lam = 2 * np.pi * 299.792458 / (prepared.w0_rad_fs / 2 + w)
        data = np.column_stack([t, np.abs(E) ** 2, np.unwrap(np.angle(E)),
                                w + prepared.w0_rad_fs / 2, lam,
                                np.abs(Ew) ** 2, np.unwrap(np.angle(Ew))])
        np.savetxt(path, data, delimiter=",", fmt="%.8g",
                   header="t_fs,intensity_t,phase_t_rad,omega_rad_fs,"
                          "wavelength_nm,intensity_w,phase_w_rad", comments="")
        return
    np.savez_compressed(
        path, kind="tadpole_result", E=E, t_fs=t, w_rad_fs=np.asarray(result["w"]),
        w0_pulse_rad_fs=prepared.w0_rad_fs / 2,
        trace_retrieved=np.asarray(result["trace"]), trace_measured=prepared.trace,
        errors_G=np.asarray(result["errors"]), errors_Gp=np.asarray(result["errors_p"]),
        G=float(result["G"]), Gp=float(result["Gp"]),
        algorithm=str(result.get("algorithm", "")),
        iterations=int(result["iterations"]), stopped=str(result["stopped"]),
        source=prepared.source, prep_settings=json.dumps(prepared.settings))
