"""The Analysis page's figure, filled once a retrieval has finished.

    ┌ Measured ──────────┬ Retrieved ───────[G]┬ |Measured − Retrieved| ┐ ║
    ├ Convergence ─┬ Pulse ───────────────┬ Spectrum ─────────────────────┤
    │ log–log G, G′│ |E(t)|², TL, φ(t)    │ vs λ, φ(λ)                    │
    └──────────────┴──────────────────────┴───────────────────────────────┘

The three maps are the same size, share their axes (a zoom on one shows the
same region in each) and ONE colour map, range and bar. The residual is
shown as |Δ| on the traces' own scale, so on a log scale it reads directly
as "how far below the signal". The bottom row has its own column widths: a
narrow convergence plot, wider pulse and spectrum, whose read-outs sit in
headroom above the curves.
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from tadpole.analysis import centred_range, pulse_views
from tadpole.design.tokens import (ERROR_COLORS, INTENSITY_COLOR, PALETTE,
                                   PHASE_COLOR)
from tadpole.gui.canvas.base import TadpoleCanvas, trace_cmap, trace_norm

BOTTOM_ROW = 0.7              # bottom strip's height vs the traces'
BOTTOM_WIDTHS = [0.6, 1, 1]   # convergence narrower than pulse and spectrum
HEADROOM = 1.45               # y range per unit of data range: room for read-outs


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
        axs = self.grid(2, 3, height_ratios=[1, BOTTOM_ROW],
                        row_widths={1: BOTTOM_WIDTHS})
        a_meas, a_ret, a_diff = axs[0]
        a_err, a_t, a_w = axs[1]
        for a in (a_ret, a_diff):
            a.sharex(a_meas)
            a.sharey(a_meas)
        self.style_axes(a_meas, "Measured", "τ (fs)", "ω (rad/fs)", grid=False)
        self.style_axes(a_ret, "Retrieved", "τ (fs)", "", grid=False)
        self.style_axes(a_diff, "|Measured − Retrieved|", "τ (fs)", "", grid=False)
        for a in (a_ret, a_diff):
            a.tick_params(labelleft=False)
        self.style_axes(a_err, "Convergence", "Iteration", "G")
        self.style_axes(a_t, "Pulse", "t (fs)", "")
        self.style_axes(a_w, "Spectrum", "λ (nm)", "")
        p, r = self.prepared, self.result
        if p is None or r is None:
            for ax in (a_meas, a_ret, a_diff, a_err, a_t, a_w):
                self.empty_message(ax, "Run a retrieval first")
            self.finish_figure()
            return

        w = p.w0_rad_fs + p.w_rad_fs
        t = p.t_fs
        ext = [t[0] - p.dt_fs / 2, t[-1] + p.dt_fs / 2,
               w[0] - p.dw_rad_fs / 2, w[-1] + p.dw_rad_fs / 2]
        kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest")
        retrieved = np.asarray(r["trace"])

        # Measured | Retrieved | |Δ|: one norm, one colour map, one bar
        norm = trace_norm(self.log_scale)
        cmap = trace_cmap(self.cmap_name, self.cmap_reversed)
        diff = p.trace - retrieved
        # Exact zeros (pixels empty in both) take the scale's floor colour, as
        # in the traces, not the background a log scale gives non-positive values.
        absdiff = np.maximum(np.abs(diff), 1e-12)
        im_m = a_meas.imshow(p.trace, cmap=cmap, norm=norm, **kw)
        im_r = a_ret.imshow(retrieved, cmap=cmap, norm=norm, **kw)
        im_d = a_diff.imshow(absdiff, cmap=cmap, norm=norm, **kw)
        cb = self.colorbar(im_m, [a_meas, a_ret, a_diff])
        self.info_box(a_ret, f"G  {r['G']:.3e}\nG′ {r['Gp']:.3e}", corner="upper right")
        m = float(np.nanmax(np.abs(diff)))
        rms = float(np.sqrt(np.nanmean(diff ** 2)))
        self.info_box(a_diff, f"max |Δ| {m:.3f}\nrms Δ   {rms:.4f}", corner="upper right")

        for key, a in (("meas", a_meas), ("ret", a_ret), ("diff", a_diff)):
            self.register_view(key, a, group="trace")
        self.register_cbar("trace_cbar", cb, [im_m, im_r, im_d])
        # centred on the grid's middle: τ = 0 and ω₀, the spectral centroid
        sl = centred_range(w, p.trace.sum(axis=1), p.w0_rad_fs)
        dl = centred_range(t, p.trace.sum(axis=0), 0.0)
        for key in ("meas", "ret", "diff"):
            self.set_auto_limits(key, dl, sl)

        self._draw_convergence(a_err, r)
        v = pulse_views(r["E"], t, p.w_rad_fs, p.w0_rad_fs)
        self._draw_pulse(a_t, v)
        self._draw_spectrum(a_w, v)
        self.finish_figure()

    def _draw_convergence(self, ax, r):
        """Log–log: decades on both axes, the final values in the legend."""
        G, Gp = np.asarray(r["errors"], float), np.asarray(r["errors_p"], float)
        it = np.arange(1, G.size + 1)
        ax.loglog(it, G, color=ERROR_COLORS[0], lw=1.3, label=f"G  {r['G']:.2e}")
        ax.loglog(it, Gp, color=ERROR_COLORS[1], lw=1.3, label=f"G′ {r['Gp']:.2e}")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(numticks=5))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlim(1, max(it[-1], 2))
        self.legend(ax, loc="upper right")
        self.register_view("err", ax)

    def _draw_pulse(self, ax, v):
        """|E(t)|² against its transform limit (the same spectrum, flat
        phase, same energy: its peak is the peak-power gain), and φ(t)."""
        t = v["t"]
        ax.plot(t, v["It_tl"], color=PALETTE["text_dim"], lw=1.1, ls="--",
                label="transform limit")
        ax.plot(t, v["It"], color=INTENSITY_COLOR, lw=1.4, label="retrieved")
        tw = ax.twinx()
        self.style_twin(tw, "φ (rad)", PHASE_COLOR)
        tw.plot(t, v["phi_t"], color=PHASE_COLOR, lw=1.0, ls=":")
        self.legend(ax, loc="upper right")
        self.info_box(ax, f"FWHM {v['fwhm_t']:.1f} fs  (TL {v['fwhm_tl']:.1f})\n"
                          f"TL peak ×{v['tl_peak']:.2f}")
        self.register_view("t", ax)
        self.register_view("tp", tw, parent="t")
        span = centred_range(t, v["It"] + v["It_tl"], 0.0, 1e-3, 0.15)
        self.set_auto_limits("t", span, (0, HEADROOM * max(1.0, v["tl_peak"])))
        self.set_auto_limits("tp", ylim=_headroom(v["phi_t"]))

    def _draw_spectrum(self, ax, v):
        lam, I = v["lam"], v["Il"]
        ax.fill_between(lam, I, color=INTENSITY_COLOR, alpha=0.25, lw=0)
        ax.plot(lam, I, color=INTENSITY_COLOR, lw=1.3)
        tw = ax.twinx()
        self.style_twin(tw, "φ (rad)", PHASE_COLOR)
        tw.plot(lam, v["phi_lam"], color=PHASE_COLOR, lw=1.3)
        self.info_box(ax, f"FWHM {v['fwhm_lam']:.1f} nm\nλ₀ {v['lambda0_nm']:.0f} nm")
        self.register_view("w", ax)
        self.register_view("wp", tw, parent="w")
        span = centred_range(lam, I, v["lambda0_nm"], 1e-3, 0.2)
        self.set_auto_limits("w", span, (0, HEADROOM))
        self.set_auto_limits("wp", ylim=_headroom(v["phi_lam"]))


def _headroom(y):
    """y limits that keep a (NaN-blanked) phase in the lower part of the
    panel, under the read-outs — the same share the intensity gets."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(y.min()), float(y.max())
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    return lo, lo + (hi - lo) * HEADROOM
