"""The Prepare page's figure.

    ┌ Raw trace ───────────┬ Prepared trace (N×N grid) ┐
    │ λ vs delay (as read) │ ω vs τ, λ on the right     │
    ├ Delay marginal ──────┼ Spectral marginal ─────────┤
    │ measured vs grid     │ measured vs grid           │
    └──────────────────────┴────────────────────────────┘
"""

from __future__ import annotations

import numpy as np

from tadpole.design.tokens import INTENSITY_COLOR, PALETTE, RETRIEVED_COLOR
from tadpole.gui.canvas.base import (TadpoleCanvas, lambda_axis, trace_cmap,
                                     trace_norm)
from tadpole.prep.grid import wavelength_to_frequency


class PrepCanvas(TadpoleCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw = None            # RawTrace
        self.processed = None      # (trace, wl, delays) after crop + background
        self.prepared = None       # PreparedTrace
        self.report = None         # PrepReport
        self.redraw()

    def set_data(self, raw=None, processed=None, prepared=None, report=None):
        self.raw, self.processed = raw, processed
        self.prepared, self.report = prepared, report
        self.redraw()

    def redraw(self):
        fig = self.fig
        fig.clear()
        fig.set_facecolor(PALETTE["plot_bg"])
        axs = fig.subplots(2, 2, height_ratios=[1.6, 1])
        a_raw, a_grid = axs[0]
        a_del, a_spec = axs[1]
        cmap = trace_cmap(self.cmap_name, self.cmap_reversed)

        # ── raw trace ──
        self.style_axes(a_raw, "Raw trace", "Delay (fs)", "Wavelength (nm)", grid=False)
        if self.processed is not None:
            T, wl, d = self.processed
            im = a_raw.imshow(T, origin="lower", aspect="auto", cmap=cmap,
                              norm=trace_norm(self.log_scale, max(T.max(), 1e-12)),
                              extent=[d[0], d[-1], wl[0], wl[-1]],
                              interpolation="nearest")
            self.style_colorbar(fig.colorbar(im, ax=a_raw, pad=0.01, fraction=0.04))
            a_raw.set_title("Raw trace (cropped, background removed)", color=PALETTE["accent"],
                            fontsize=11, fontweight="bold", loc="left")
        else:
            self.empty_message(a_raw, "Open a .dwc file (or Simulate)")

        # ── prepared grid trace ──
        self.style_axes(a_grid, "Prepared trace", "Delay τ (fs)",
                        "Signal frequency ω (rad/fs)", grid=False)
        p = self.prepared
        if p is not None:
            w = p.w0_rad_fs + p.w_rad_fs
            t = p.t_fs
            half_t, half_w = p.dt_fs / 2, p.dw_rad_fs / 2
            im = a_grid.imshow(p.trace, origin="lower", aspect="auto", cmap=cmap,
                               norm=trace_norm(self.log_scale),
                               extent=[t[0] - half_t, t[-1] + half_t,
                                       w[0] - half_w, w[-1] + half_w],
                               interpolation="nearest")
            sec = a_grid.secondary_yaxis("right", functions=(lambda_axis, lambda_axis))
            self.style_secondary(sec, "λ (nm)")
            self.style_colorbar(fig.colorbar(im, ax=a_grid, pad=0.08, fraction=0.04))
            a_grid.set_title(f"Prepared trace  {p.n}×{p.n},  dt = {p.dt_fs:.3g} fs",
                             color=PALETTE["accent"], fontsize=11, fontweight="bold",
                             loc="left")
        else:
            self.empty_message(a_grid, "No prepared trace")

        # ── marginals ──
        self.style_axes(a_del, "Delay marginal", "Delay τ (fs)", "Norm.")
        self.style_axes(a_spec, "Spectral marginal", "Signal frequency ω (rad/fs)", "Norm.")
        if self.processed is not None:
            T, wl, d = self.processed
            Tw, w = wavelength_to_frequency(T, wl)
            dm = Tw.sum(axis=0)
            shift = self.report.zero_shift_fs if self.report is not None else 0.0
            if dm.max() > 0:
                a_del.plot(d - shift, dm / dm.max(), color=INTENSITY_COLOR, lw=1.2,
                           label="measured")
            sm = Tw.sum(axis=1)
            if sm.max() > 0:
                a_spec.plot(w, sm / sm.max(), color=INTENSITY_COLOR, lw=1.0,
                            label="measured")
        if p is not None:
            gd = p.trace.sum(axis=0)
            gs = p.trace.sum(axis=1)
            if gd.max() > 0:
                a_del.plot(p.t_fs, gd / gd.max(), color=RETRIEVED_COLOR, lw=1.2,
                           ls="--", label="on grid")
            if gs.max() > 0:
                a_spec.plot(p.w0_rad_fs + p.w_rad_fs, gs / gs.max(),
                            color=RETRIEVED_COLOR, lw=1.2, ls="--", label="on grid")
            for ax, lo, hi in ((a_del, p.t_fs[0], p.t_fs[-1]),
                               (a_spec, p.w0_rad_fs + p.w_rad_fs[0],
                                p.w0_rad_fs + p.w_rad_fs[-1])):
                ax.axvspan(lo, hi, color=PALETTE["accent"], alpha=0.07, lw=0)
            sec = a_spec.secondary_xaxis("top", functions=(lambda_axis, lambda_axis))
            self.style_secondary(sec, "λ (nm)", which="x")
        for ax in (a_del, a_spec):
            if ax.lines:
                self.legend(ax, loc="upper right")
        if self.report is not None:
            r = self.report
            self.info_box(a_del,
                          f"window {r.delay_window_fs:.0f} fs\n"
                          f"signal {r.delay_span_fs:.0f} fs\n"
                          f"zero  {r.zero_shift_fs:+.1f} fs\n"
                          f"asym  {100*r.asymmetry:.1f}%")
            self.info_box(a_spec,
                          f"window {r.freq_window_rad_fs:.3f}\n"
                          f"signal {r.freq_span_rad_fs:.3f}\n"
                          f"energy {100*r.energy_inside:.1f}%")
        self.draw_idle()
