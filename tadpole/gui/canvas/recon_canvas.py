"""The Reconstruct page's figure, updated live while Julia iterates.

    ┌ Measured ───────┬ Retrieved ──────┬ Convergence (G, G′) ┐
    ├ Temporal ───────┼ Spectral ───────┼ Difference ──────────┤
    │ |E(t)|², φ(t)   │ |E(ω)|², φ(ω)   │ measured − retrieved │
    └─────────────────┴─────────────────┴──────────────────────┘

`redraw()` builds every artist; `show_progress()` only moves data into them,
so a snapshot costs one canvas draw and no figure rebuild.
"""

from __future__ import annotations

import numpy as np

from tadpole.analysis import pulse_views, visible_range
from tadpole.design.tokens import (ERROR_COLORS, INTENSITY_COLOR, PALETTE,
                                   PHASE_COLOR)
from tadpole.gui.canvas.base import (TadpoleCanvas, lambda_axis, trace_cmap,
                                     trace_norm)
from matplotlib.colors import Normalize


class ReconCanvas(TadpoleCanvas):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.prepared = None
        self.trace = None          # latest retrieved trace
        self.E = None              # latest field
        self.iters: list[int] = []
        self.G: list[float] = []
        self.Gp: list[float] = []
        self._art = {}
        self.redraw()

    # ── data in ──────────────────────────────────────────────────────────
    def set_prepared(self, prepared):
        self.prepared = prepared
        self.reset_progress()

    def reset_progress(self):
        self.trace = None
        self.E = None
        self.iters, self.G, self.Gp = [], [], []
        self.redraw()

    def show_progress(self, iteration, G, Gp, trace, E):
        self.iters.append(iteration); self.G.append(G); self.Gp.append(Gp)
        self.trace, self.E = trace, E
        if "ret" not in self._art:
            self.redraw()
            return
        self._update_artists()
        self.draw_idle()

    def show_result(self, result):
        """Final result: the full error history replaces the sampled one."""
        n = len(result["errors"])
        self.iters = list(range(1, n + 1))
        self.G = list(result["errors"])
        self.Gp = list(result["errors_p"])
        self.trace, self.E = result["trace"], result["E"]
        self.redraw()

    # ── drawing ──────────────────────────────────────────────────────────
    def redraw(self):
        fig = self.fig
        fig.clear()
        fig.set_facecolor(PALETTE["plot_bg"])
        self._art = {}
        axs = fig.subplots(2, 3)
        (a_meas, a_ret, a_err), (a_t, a_w, a_diff) = axs
        self._axes = dict(meas=a_meas, ret=a_ret, err=a_err, t=a_t, w=a_w, diff=a_diff)
        p = self.prepared
        cmap = trace_cmap(self.cmap_name, self.cmap_reversed)

        self.style_axes(a_meas, "Measured", "Delay τ (fs)", "ω (rad/fs)", grid=False)
        self.style_axes(a_ret, "Retrieved", "Delay τ (fs)", "", grid=False)
        self.style_axes(a_err, "Convergence", "Iteration", "Error")
        self.style_axes(a_t, "Temporal", "Time (fs)", "Intensity (norm.)")
        self.style_axes(a_w, "Spectral (fundamental)", "ω (rad/fs)", "Intensity (norm.)")
        self.style_axes(a_diff, "Difference", "Delay τ (fs)", "", grid=False)

        if p is None:
            for ax in (a_meas, a_ret, a_t, a_w, a_diff):
                self.empty_message(ax, "Prepare a trace first")
            self.draw_idle()
            return

        w = p.w0_rad_fs + p.w_rad_fs
        t = p.t_fs
        ext = [t[0] - p.dt_fs / 2, t[-1] + p.dt_fs / 2,
               w[0] - p.dw_rad_fs / 2, w[-1] + p.dw_rad_fs / 2]
        kw = dict(origin="lower", aspect="auto", cmap=cmap, extent=ext,
                  interpolation="nearest", norm=trace_norm(self.log_scale))
        im = a_meas.imshow(p.trace, **kw)
        self.style_colorbar(fig.colorbar(im, ax=a_meas, pad=0.01, fraction=0.05))
        # Before the first snapshot both panels are blank (NaN shows as the
        # plot background), not a misleading "everything differs" image.
        ret = self.trace if self.trace is not None else np.full_like(p.trace, np.nan)
        self._art["ret"] = a_ret.imshow(ret, **kw)
        self.style_colorbar(fig.colorbar(self._art["ret"], ax=a_ret, pad=0.01,
                                         fraction=0.05))
        self._art["diff"] = a_diff.imshow(p.trace - ret, origin="lower", aspect="auto",
                                          cmap="RdBu_r", extent=ext,
                                          interpolation="nearest",
                                          norm=Normalize(-0.1, 0.1))
        self.style_colorbar(fig.colorbar(self._art["diff"], ax=a_diff, pad=0.01,
                                         fraction=0.05))
        # zoom both traces onto where the signal is
        sl = visible_range(w, p.trace.sum(axis=1), rel=1e-3, pad=0.1)
        dl = visible_range(t, p.trace.sum(axis=0), rel=1e-3, pad=0.1)
        for ax in (a_meas, a_ret, a_diff):
            ax.set_ylim(*sl); ax.set_xlim(*dl)

        # convergence
        self._art["G"], = a_err.semilogy(self.iters or [np.nan], self.G or [np.nan],
                                         color=ERROR_COLORS[0], lw=1.3, label="G")
        self._art["Gp"], = a_err.semilogy(self.iters or [np.nan], self.Gp or [np.nan],
                                          color=ERROR_COLORS[1], lw=1.3, label="G′")
        self.legend(a_err, loc="upper right")
        self._art["err_text"] = self.info_box(a_err, "")

        # temporal / spectral with phase twins
        self._art["It"], = a_t.plot(t, np.full_like(t, np.nan), color=INTENSITY_COLOR, lw=1.4)
        a_tp = a_t.twinx()
        self.style_twin(a_tp, "Phase (rad)", PHASE_COLOR)
        self._art["phi_t"], = a_tp.plot(t, np.full_like(t, np.nan), color=PHASE_COLOR,
                                        lw=1.1, ls="--")
        wp = p.w0_rad_fs / 2 + p.w_rad_fs
        self._art["Iw"], = a_w.plot(wp, np.full_like(wp, np.nan), color=INTENSITY_COLOR,
                                    lw=1.4)
        a_wp = a_w.twinx()
        self.style_twin(a_wp, "Phase (rad)", PHASE_COLOR)
        self._art["phi_w"], = a_wp.plot(wp, np.full_like(wp, np.nan), color=PHASE_COLOR,
                                        lw=1.1, ls="--")
        sec = a_w.secondary_xaxis("top", functions=(lambda_axis, lambda_axis))
        self.style_secondary(sec, "λ (nm)", which="x")
        self._axes.update(tp=a_tp, wp=a_wp)
        self._art["t_text"] = self.info_box(a_t, "")
        self._art["w_text"] = self.info_box(a_w, "")
        for ax in (a_t, a_w):
            ax.set_ylim(0, 1.05)
        # sensible empty view until a field arrives
        a_t.set_xlim(dl[0] / 2, dl[1] / 2)
        a_w.set_xlim(p.w0_rad_fs / 2 + (sl[0] - p.w0_rad_fs) / 2,
                     p.w0_rad_fs / 2 + (sl[1] - p.w0_rad_fs) / 2)
        self._update_artists()
        self.draw_idle()

    def _update_artists(self):
        p = self.prepared
        if p is None or "ret" not in self._art:
            return
        a = self._art
        if self.trace is not None:
            a["ret"].set_data(self.trace)
            a["diff"].set_data(p.trace - self.trace)
        if self.G:
            a["G"].set_data(self.iters, self.G)
            a["Gp"].set_data(self.iters, self.Gp)
            ax = self._axes["err"]
            ax.relim(); ax.autoscale_view()
            a["err_text"].set_text(f"G  = {self.G[-1]:.3e}\nG′ = {self.Gp[-1]:.3e}\n"
                                   f"iter {self.iters[-1]}")
        if self.E is not None:
            v = pulse_views(self.E, p.t_fs, p.w_rad_fs, p.w0_rad_fs)
            a["It"].set_ydata(v["It"]); a["phi_t"].set_ydata(v["phi_t"])
            a["Iw"].set_ydata(v["Iw"]); a["phi_w"].set_ydata(v["phi_w"])
            self._axes["t"].set_xlim(*visible_range(v["t"], v["It"], 1e-3, 0.2))
            self._axes["w"].set_xlim(*visible_range(v["w"], v["Iw"], 1e-3, 0.2))
            for key in ("tp", "wp"):
                ax = self._axes[key]
                ax.relim(); ax.autoscale_view(scalex=False)
            a["t_text"].set_text(f"FWHM {v['fwhm_t']:.1f} fs")
            a["w_text"].set_text(f"FWHM {v['fwhm_w']:.3f} rad/fs\n"
                                 f"λ₀ {v['lambda0_nm']:.0f} nm")
