"""The Propagator page's figure.

    ┌ After — as retrieved ─┬ After — time-reversed ─┬ Measured after ────┐ ║
    ├ Temporal ─────────────┼ Spectral ──────────────┼ Direction scan ─────┤
    │ before / both afters  │ |E(ω)|², phase before/ │ G vs s·stack, ±1    │
    │                       │ after, λ on top        │ marked, best fit    │
    └───────────────────────┴────────────────────────┴─────────────────────┘

The three traces share their axes and one colour range. Everything shown is
computed by the controller (controllers/propagator.py) and handed over as a
`PropView`; this module only draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tadpole.analysis import centred_range, pulse_views
from tadpole.design.tokens import (ERROR_COLORS, INTENSITY_COLOR, PALETTE,
                                   PHASE_COLOR, RETRIEVED_COLOR)
from tadpole.gui.canvas.base import TadpoleCanvas, trace_cmap, trace_norm


@dataclass
class PropView:
    pulse: object                 # tadpole.io.prepared.PulseSource
    E_as: np.ndarray              # after the stack, as retrieved
    E_rev: np.ndarray             # after the stack, time-reversed
    T_as: np.ndarray              # their SHG traces (peak 1)
    T_rev: np.ndarray
    gdd: float                    # the stack's GDD at the pulse centre (fs²)
    meas: object = None           # PreparedTrace on the pulse grid, or None
    G_as: float | None = None
    G_rev: float | None = None
    scan: tuple | None = None     # (s, G, best_s)
    scanning: bool = False


class PropagatorCanvas(TadpoleCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view: PropView | None = None
        self.status = "Run a retrieval or load a saved result"
        self.redraw()

    def set_view(self, view: PropView | None, status: str = ""):
        old = self.view
        if view is None or old is None or view.pulse is not old.pulse:
            self.clear_manual()    # another pulse (grid) starts from its own view
        self.view = view
        if status:
            self.status = status
        self.redraw()

    # ── drawing ──────────────────────────────────────────────────────────
    def redraw(self):
        self.begin_figure()
        axs = self.grid(2, 3, height_ratios=[1.25, 1])
        (a_as, a_rev, a_meas), (a_t, a_w, a_s) = axs
        for a in (a_rev, a_meas):
            a.sharex(a_as)
            a.sharey(a_as)
        v = self.view
        self.style_axes(a_as, "After — as retrieved", "τ (fs)", "ω (rad/fs)", grid=False)
        self.style_axes(a_rev, "After — time-reversed", "τ (fs)", "", grid=False)
        self.style_axes(a_meas, "Measured after", "τ (fs)", "", grid=False)
        for a in (a_rev, a_meas):
            a.tick_params(labelleft=False)
        self.style_axes(a_t, "Temporal", "t (fs)", "I (norm.)")
        self.style_axes(a_w, "Spectral", "λ (nm)", "")
        self.style_axes(a_s, "Direction scan", "s  (× stack dispersion)", "G")
        if v is None:
            for a in (a_as, a_rev, a_meas, a_t, a_w, a_s):
                self.empty_message(a, self.status)
            self.finish_figure()
            return

        p = v.pulse
        wsig = p.w0_signal_rad_fs + p.w_rad_fs
        t = p.t_fs
        dt, dw = p.dt_fs, p.w_rad_fs[1] - p.w_rad_fs[0]
        ext = [t[0] - dt / 2, t[-1] + dt / 2, wsig[0] - dw / 2, wsig[-1] + dw / 2]
        norm = trace_norm(self.log_scale)
        kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest",
                  cmap=trace_cmap(self.cmap_name, self.cmap_reversed), norm=norm)
        ims = [a_as.imshow(v.T_as, **kw), a_rev.imshow(v.T_rev, **kw)]
        ref = v.T_as
        if v.meas is not None:
            ims.append(a_meas.imshow(np.where(v.meas.mask, v.meas.trace, np.nan), **kw))
            ref = v.meas.trace
        else:
            self.empty_message(a_meas, "Load the trace measured\nafter the stack")
        cb = self.colorbar(ims[0], [a_as, a_rev, a_meas])
        for key, a in (("as", a_as), ("rev", a_rev), ("meas", a_meas)):
            self.register_view(key, a, group="trace")
        self.register_cbar("trace_cbar", cb, ims)
        sl = centred_range(wsig, ref.sum(axis=1), p.w0_signal_rad_fs)
        dl = centred_range(t, ref.sum(axis=0), 0.0)
        for key in ("as", "rev", "meas"):
            self.set_auto_limits(key, dl, sl)
        if v.G_as is not None:
            self.info_box(a_as, f"G {v.G_as:.3e}", corner="upper right")
            self.info_box(a_rev, f"G {v.G_rev:.3e}", corner="upper right")

        self._draw_temporal(a_t, v)
        self._draw_spectral(a_w, v)
        self._draw_scan(a_s, v)
        self.finish_figure()

    def _draw_temporal(self, ax, v):
        p = v.pulse
        t = p.t_fs
        before = pulse_views(p.E, t, p.w_rad_fs, p.w0_signal_rad_fs)
        after = pulse_views(v.E_as, t, p.w_rad_fs, p.w0_signal_rad_fs)
        rev = pulse_views(v.E_rev, t, p.w_rad_fs, p.w0_signal_rad_fs)
        ax.plot(t, before["It"], color=PALETTE["text_dim"], lw=1.1, ls="--", label="before")
        ax.plot(t, after["It"], color=INTENSITY_COLOR, lw=1.4, label="after (as retr.)")
        ax.plot(t, rev["It"], color=RETRIEVED_COLOR, lw=1.2, label="after (reversed)")
        self.legend(ax, loc="upper right")
        self.info_box(ax, f"FWHM {before['fwhm_t']:.1f} →\n{after['fwhm_t']:.1f}"
                          f" / {rev['fwhm_t']:.1f} fs")
        span = centred_range(t, after["It"] + rev["It"] + before["It"], 0.0, 1e-3, 0.15)
        self.register_view("t", ax)
        self.set_auto_limits("t", span, (0, 1.12))

    def _draw_spectral(self, ax, v):
        p = v.pulse
        before = pulse_views(p.E, p.t_fs, p.w_rad_fs, p.w0_signal_rad_fs)
        after = pulse_views(v.E_as, p.t_fs, p.w_rad_fs, p.w0_signal_rad_fs)
        lam, I = before["lam"], before["Il"]
        ax.fill_between(lam, I, color=INTENSITY_COLOR, alpha=0.25, lw=0)
        ax.plot(lam, I, color=INTENSITY_COLOR, lw=1.3)
        tw = ax.twinx()
        self.style_twin(tw, "φ (rad)", PHASE_COLOR)
        tw.plot(lam, before["phi_lam"], color=PHASE_COLOR, lw=1.1, ls="--", label="before")
        tw.plot(lam, after["phi_lam"], color=PHASE_COLOR, lw=1.4, label="after")
        self.legend(tw, loc="upper right")
        self.register_view("w", ax)
        self.register_view("wp", tw, parent="w")
        lo, hi = centred_range(lam, I, before["lambda0_nm"], 1e-3, 0.2)
        self.set_auto_limits("w", (lo, hi), (0, 1.08))

    def _draw_scan(self, ax, v):
        self.register_view("scan", ax)
        if v.meas is None:
            self.empty_message(ax, "Load a measured trace\nto scan")
            return
        if v.scan is None:
            self.empty_message(ax, "scanning…" if v.scanning else "")
            return
        s, G, best = v.scan
        lo, hi = float(s[0]), float(s[-1])
        ax.axvspan(lo, 0, color=PALETTE["danger"], alpha=0.07, lw=0)
        ax.axvspan(0, hi, color=PALETTE["good"], alpha=0.07, lw=0)
        for x in (-1, 1):
            ax.axvline(x, color=PALETTE["text_dim"], lw=0.9, ls=":")
        ax.plot(s, G, color=ERROR_COLORS[0], lw=1.4)
        ax.axvline(best, color=PALETTE["accent"], lw=1.2)
        ax.text(0.02, 0.97, "reversed", transform=ax.transAxes, ha="left", va="top",
                color=PALETTE["danger"], fontsize=8)
        ax.text(0.98, 0.97, "as retrieved", transform=ax.transAxes, ha="right",
                va="top", color=PALETTE["good"], fontsize=8)
        self.info_box(ax, f"best s = {best:+.2f}", corner="upper center")
        if abs(v.gdd) > 1e-9:
            sec = ax.secondary_xaxis("top", functions=(lambda x: x * v.gdd,
                                                       lambda g: g / v.gdd))
            self._ticks(sec)
            for sp in sec.spines.values():
                sp.set_edgecolor(PALETTE["border"])
            sec.set_xlabel("GDD (fs²)", color=PALETTE["text_dim"], fontsize=7.5,
                           labelpad=1, loc="right")
        pad = 0.05 * (np.nanmax(G) - np.nanmin(G) or 1.0)
        self.set_auto_limits("scan", (lo, hi), (np.nanmin(G) - pad, np.nanmax(G) + pad))
