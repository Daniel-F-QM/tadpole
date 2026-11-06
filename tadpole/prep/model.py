"""The data that flows from the Prepare page to the Reconstruct page.

No Qt, no Julia: plain dataclasses of numpy arrays, so a prepared trace can be
tested, saved (.npz, see tadpole.io.prepared) and handed to the engine as is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

C_NM_PER_FS = 299.792458           # speed of light


@dataclass
class PrepSettings:
    """Every knob of the preparation pipeline (tadpole.prep.pipeline.prepare).

    bg_mode       "none" | "constant" | "spectral"
                    constant: subtract one number, the mean of the edge columns
                    spectral: subtract, per wavelength, the mean of the edge
                              delay columns (a background spectrum)
    bg_edge_cols  how many delay columns at EACH end count as "edge"
    bg_edge_rows  additionally subtract, per delay, the mean of this many
                  wavelength rows at each end (0 = off)
    wl_min_nm…    crop window; None = keep the full axis
    delay_min_fs… crop window in measured (file-centred) delays
    zero_mode     "centroid" | "peak" | "none": how delay zero is found
    zero_offset_fs extra manual shift added on top of zero_mode
    symmetrize    average T(τ) and T(−τ) on the grid (SHG traces are
                  symmetric; this halves uncorrelated noise)
    grid_n        N of the N×N Fourier grid
    grid_dt_fs    time step; 0 = automatic (balances delay and frequency
                  coverage, see grid.auto_dt)
    """
    bg_mode: str = "spectral"
    bg_edge_cols: int = 5
    bg_edge_rows: int = 0
    wl_min_nm: float | None = None
    wl_max_nm: float | None = None
    delay_min_fs: float | None = None
    delay_max_fs: float | None = None
    zero_mode: str = "centroid"
    zero_offset_fs: float = 0.0
    symmetrize: bool = False
    grid_n: int = 128
    grid_dt_fs: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedTrace:
    """A trace on a Fourier-consistent grid, ready for retrieval.

    Grid (both GPA/PCGPA):
      trace[ω_i, τ_k]  N×N, peak 1. Rows: signal frequency offsets
                       w = (k − N/2)·dω around w0. Columns: delays
                       τ = (k − N/2)·dt — the same values as the time grid.
      mask             N×N bool, True where the pixel lies inside the measured
                       (and cropped) data; outside it was filled with 0.

    Measured delays (for the ptychographic algorithm, which does not need a
    square grid):
      trace_meas[ω_i, j]  N×J: resampled in frequency only, at the J measured
                          (zero-shifted, cropped) delays `delays_meas_fs`
      mask_meas           N×J bool
    """
    trace: np.ndarray
    mask: np.ndarray
    dt_fs: float
    w0_rad_fs: float                 # absolute centre (SIGNAL) angular frequency
    trace_meas: np.ndarray
    mask_meas: np.ndarray
    delays_meas_fs: np.ndarray
    source: str = ""
    settings: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.trace.shape[0]

    @property
    def dw_rad_fs(self) -> float:
        return 2 * np.pi / (self.n * self.dt_fs)

    @property
    def t_fs(self) -> np.ndarray:
        return (np.arange(self.n) - self.n // 2) * self.dt_fs

    @property
    def w_rad_fs(self) -> np.ndarray:
        """Frequency OFFSETS of the trace rows."""
        return (np.arange(self.n) - self.n // 2) * self.dw_rad_fs

    @property
    def lambda0_nm(self) -> float:
        """Centre wavelength of the SIGNAL (the SHG light)."""
        return 2 * np.pi * C_NM_PER_FS / self.w0_rad_fs

    @property
    def wavelengths_nm(self) -> np.ndarray:
        """Signal wavelength of each trace row (descending, since ω ascends)."""
        return 2 * np.pi * C_NM_PER_FS / (self.w0_rad_fs + self.w_rad_fs)
