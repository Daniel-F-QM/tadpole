"""Reading and writing .dwc FROG traces (the Femtosoft layout Lillypad writes).

    Number of delay points = 800          <- trace columns
    Number of wavelength points = 2068    <- trace rows
    Delay increment = 1.0000000           <- fs, uniform; no absolute axis
    <blank>
    [Wavelength vector]
    <n_wl tab-separated nm values>
    <blank>
    [Data array]
    <n_wl rows of n_delays tab-separated counts>

Tab-separated, CRLF, LabVIEW "%.6E" numbers with an unpadded exponent. The
format stores NO absolute delay: the reader centres the axis on the middle
column, and the preparation step decides where zero really is.

The writer is Lillypad's (lillypad/scan/writers.py), so a trace saved here
loads into Femtosoft and back into Lillypad unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HEADER_RE = {
    "n_delays": re.compile(r"Number of delay points\s*=\s*(\d+)", re.I),
    "n_wl": re.compile(r"Number of wavelength points\s*=\s*(\d+)", re.I),
    "dtau": re.compile(r"Delay increment\s*=\s*([-+0-9.eE]+)", re.I),
}


@dataclass
class RawTrace:
    """A measured trace as loaded: rows are wavelength, columns are delay.

    `delays_fs` is centred on the middle column (the file has no absolute
    zero); `source` is the file it came from, for titles and metadata.
    """
    trace: np.ndarray            # (n_wl, n_delays), float64
    wavelengths_nm: np.ndarray   # (n_wl,)
    delays_fs: np.ndarray        # (n_delays,)
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def delay_step_fs(self) -> float:
        d = np.diff(self.delays_fs)
        return float(np.median(np.abs(d))) if d.size else 0.0


def centred_delays(n: int, step: float) -> np.ndarray:
    """n delays `step` apart, with the middle column at 0 (index n//2)."""
    return (np.arange(n) - n // 2) * float(step)


def read_dwc(path) -> RawTrace:
    """Load a .dwc file. Raises ValueError with a readable message on a
    malformed file."""
    path = Path(path)
    text = path.read_text(encoding="ascii", errors="replace")
    head, sep, rest = text.partition("[Wavelength vector]")
    if not sep:
        raise ValueError(f"{path.name}: no [Wavelength vector] section")
    vals = {}
    for key, rx in _HEADER_RE.items():
        m = rx.search(head)
        if not m:
            raise ValueError(f"{path.name}: header line for {key} missing")
        vals[key] = m.group(1)
    n_delays, n_wl = int(vals["n_delays"]), int(vals["n_wl"])
    dtau = float(vals["dtau"])

    wl_text, sep, data_text = rest.partition("[Data array]")
    if not sep:
        raise ValueError(f"{path.name}: no [Data array] section")
    wl = np.array(wl_text.split(), dtype=float)
    data = np.array(data_text.split(), dtype=float)
    if wl.size != n_wl:
        raise ValueError(f"{path.name}: header says {n_wl} wavelengths, "
                         f"file has {wl.size}")
    if data.size != n_wl * n_delays:
        raise ValueError(f"{path.name}: expected {n_wl}×{n_delays} values, "
                         f"found {data.size}")
    trace = data.reshape(n_wl, n_delays)
    return RawTrace(trace=trace, wavelengths_nm=wl,
                    delays_fs=centred_delays(n_delays, dtau),
                    source=str(path), meta={"delay_increment_fs": dtau})


# ── Writer (verbatim behaviour of Lillypad's write_dwc) ─────────────────────

def _row_fmt(n, spec="%.6E", sep="\t"):
    return sep.join([spec] * n)


def _lv_row(values, fmt=None):
    """One line of LabVIEW-style %.6E numbers: exponent without the padding
    zero (1.843471E+2, not 1.843471E+02)."""
    line = (fmt or _row_fmt(len(values))) % tuple(values)
    return line.replace("E+0", "E+").replace("E-0", "E-")


def write_dwc(path, trace, wavelengths_nm, delay_step_fs) -> None:
    """Write `trace` (n_wl × n_delays, ascending delay) as a .dwc file."""
    trace = np.asarray(trace, float)
    wl = np.asarray(wavelengths_nm, float)
    n_wl, n_delays = trace.shape
    if wl.size != n_wl:
        raise ValueError("wavelength vector does not match the trace rows")
    with open(path, "w", encoding="ascii", newline="\r\n") as f:
        f.write(f"Number of delay points = {n_delays}\n"
                f"Number of wavelength points = {n_wl}\n"
                f"Delay increment = {float(delay_step_fs):.7f}\n"
                f"\n[Wavelength vector]\n")
        f.write(_lv_row(wl))
        f.write("\n\n[Data array]\n")
        fmt = _row_fmt(n_delays)
        f.write("\n".join(_lv_row(row, fmt) for row in trace))
        f.write("\n")
