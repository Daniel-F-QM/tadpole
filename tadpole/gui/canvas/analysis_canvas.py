"""The Analysis page's figure, filled once a retrieval has finished.

    ┌ Convergence ───────────┬ Difference ────────────────┐ ║ ±
    │ G, G′ vs iteration     │ measured − retrieved       │
    └────────────────────────┴────────────────────────────┘
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize

from tadpole.analysis import centred_range
from tadpole.design.tokens import DARK_PALETTE, ERROR_COLORS, PALETTE
from tadpole.gui.canvas.base import TadpoleCanvas


class AnalysisCanvas(TadpoleCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.prepared = None
        self.result = None
        self.redraw()

    def set_result(self, prepared, result):
        if prepared is not self.prepared:
            self.clear_manual()
        self.prepared, self.result = prepared, result
        self.redraw()

    def clear(self):
        self.prepared = self.result = None
        self.clear_manual()
        self.redraw()

    def redraw(self):
        self.begin_figure()
        a_err, a_diff = self.grid(1, 2)[0]
        self.style_axes(a_err, "Convergence", "Iteration", "Error")
        self.style_axes(a_diff, "Difference (measured − retrieved)", "τ (fs)",
                        "ω (rad/fs)", grid=False)
        p, r = self.prepared, self.result
        if p is None or r is None:
            for ax in (a_err, a_diff):
                self.empty_message(ax, "Run a retrieval first")
            self.finish_figure()
            return

        G, Gp = np.asarray(r["errors"], float), np.asarray(r["errors_p"], float)
        it = np.arange(1, G.size + 1)
        a_err.semilogy(it, G, color=ERROR_COLORS[0], lw=1.3, label="G")
        a_err.semilogy(it, Gp, color=ERROR_COLORS[1], lw=1.3, label="G′")
        self.legend(a_err, loc="upper right")
        self.info_box(a_err, f"G  = {r['G']:.3e}\nG′ = {r['Gp']:.3e}\n"
                             f"iter {r['iterations']}")
        self.register_view("err", a_err)

        w = p.w0_rad_fs + p.w_rad_fs
        t = p.t_fs
        ext = [t[0] - p.dt_fs / 2, t[-1] + p.dt_fs / 2,
               w[0] - p.dw_rad_fs / 2, w[-1] + p.dw_rad_fs / 2]
        diff = p.trace - np.asarray(r["trace"])
        m = float(np.nanmax(np.abs(diff))) or 1.0
        # Zero takes the plot background's lightness in either theme: a dark
        # centre on the dark theme, so "no residual" is not a white glare.
        cmap = "berlin" if PALETTE["bg"] == DARK_PALETTE["bg"] else "RdBu_r"
        im = a_diff.imshow(diff, origin="lower", aspect="auto", cmap=cmap,
                           extent=ext, interpolation="nearest", norm=Normalize(-m, m))
        cb = self.colorbar(im, a_diff)
        self.register_view("diff", a_diff)
        self.register_cbar("diff_cbar", cb, [im])
        # centred on the grid's middle: τ = 0 and ω₀, the spectral centroid
        sl = centred_range(w, p.trace.sum(axis=1), p.w0_rad_fs)
        dl = centred_range(t, p.trace.sum(axis=0), 0.0)
        self.set_auto_limits("diff", dl, sl)
        rms = float(np.sqrt(np.nanmean(diff ** 2)))
        self.info_box(a_diff, f"max |Δ| {m:.3f}\nrms Δ   {rms:.4f}")
        self.finish_figure()
