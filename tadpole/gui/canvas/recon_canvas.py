"""The Reconstruct page's figure, updated live while Julia iterates.

    ┌ Measured ─────────────┬ Retrieved ─────────[G box]┐ ║ one colour bar
    ├ Temporal ─────────────┼ Spectral ─────────────────┤
    │ |E(t)|², φ(t)         │ |E(ω)|², φ(ω), λ on top   │
    └───────────────────────┴───────────────────────────┘

Measured and Retrieved share their axes AND one norm, so a zoom or a colour
range set on either applies to both. Convergence and the difference map live
on the Analysis page (analysis_canvas.py).

`redraw()` builds every artist; `show_progress()` only moves data into them,
so a snapshot costs one canvas draw and no figure rebuild.
"""

from __future__ import annotations

import numpy as np

from tadpole.analysis import centred_range, pulse_views, visible_range
from tadpole.design.tokens import INTENSITY_COLOR, PHASE_COLOR
from tadpole.gui.canvas.base import TadpoleCanvas, trace_cmap, trace_norm


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
        self._axes = {}
        self.redraw()

    # ── data in ──────────────────────────────────────────────────────────
    def set_prepared(self, prepared):
        self.prepared = prepared
        self.clear_manual()        # a new trace starts from its own view
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
        self.begin_figure()
        fig = self.fig
        self._art = {}
        axs = self.grid(2, 2, height_ratios=[1.45, 1])
        (a_meas, a_ret), (a_t, a_w) = axs
        a_ret.sharex(a_meas)
        a_ret.sharey(a_meas)
        self._axes = dict(meas=a_meas, ret=a_ret, t=a_t, w=a_w)
        p = self.prepared

        self.style_axes(a_meas, "Measured", "τ (fs)", "ω (rad/fs)", grid=False)
        self.style_axes(a_ret, "Retrieved", "τ (fs)", "", grid=False)
        a_ret.tick_params(labelleft=False)
        self.style_axes(a_t, "Temporal", "t (fs)", "I (norm.)")
        self.style_axes(a_w, "Spectral (fundamental)", "ω (rad/fs)", "")

        if p is None:
            for ax in (a_meas, a_ret, a_t, a_w):
                self.empty_message(ax, "Prepare a trace first")
            self.finish_figure()
            return

        w = p.w0_rad_fs + p.w_rad_fs
        t = p.t_fs
        ext = [t[0] - p.dt_fs / 2, t[-1] + p.dt_fs / 2,
               w[0] - p.dw_rad_fs / 2, w[-1] + p.dw_rad_fs / 2]
        norm = trace_norm(self.log_scale)          # ONE norm: one colour range
        kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest",
                  cmap=trace_cmap(self.cmap_name, self.cmap_reversed), norm=norm)
        im_meas = a_meas.imshow(p.trace, **kw)
        # Before the first snapshot the panel is blank (NaN shows as the plot
        # background), not a misleading copy of the colormap's floor.
        ret = self.trace if self.trace is not None else np.full_like(p.trace, np.nan)
        self._art["ret"] = a_ret.imshow(ret, **kw)
        cb = self.colorbar(im_meas, [a_meas, a_ret])
        self.register_view("meas", a_meas, group="trace")
        self.register_view("ret", a_ret, group="trace")
        self.register_cbar("trace", cb, [im_meas, self._art["ret"]])

        # zoom both traces onto where the signal is
        # centred on the grid's middle: τ = 0 and ω₀, the spectral centroid
        sl = centred_range(w, p.trace.sum(axis=1), p.w0_rad_fs)
        dl = centred_range(t, p.trace.sum(axis=0), 0.0)
        self.set_auto_limits("meas", dl, sl)
        self.set_auto_limits("ret", dl, sl)
        self._art["g_text"] = self.info_box(a_ret, "")
        self._art["g_text"].set_visible(False)

        # temporal / spectral with phase twins
        self._art["It"], = a_t.plot(t, np.full_like(t, np.nan), color=INTENSITY_COLOR, lw=1.4)
        a_tp = a_t.twinx()
        self.style_twin(a_tp, "φ (rad)", PHASE_COLOR)
        self._art["phi_t"], = a_tp.plot(t, np.full_like(t, np.nan), color=PHASE_COLOR,
                                        lw=1.1, ls="--")
        wp = p.w0_rad_fs / 2 + p.w_rad_fs
        self._art["Iw"], = a_w.plot(wp, np.full_like(wp, np.nan), color=INTENSITY_COLOR,
                                    lw=1.4)
        a_wp = a_w.twinx()
        self.style_twin(a_wp, "φ (rad)", PHASE_COLOR)
        self._art["phi_w"], = a_wp.plot(wp, np.full_like(wp, np.nan), color=PHASE_COLOR,
                                        lw=1.1, ls="--")
        self.lambda_secondary(a_w, "x")
        self._axes.update(tp=a_tp, wp=a_wp)
        self._art["t_text"] = self.info_box(a_tp, "")
        self._art["w_text"] = self.info_box(a_wp, "")
        self.register_view("t", a_t)
        self.register_view("tp", a_tp, parent="t")
        self.register_view("w", a_w)
        self.register_view("wp", a_wp, parent="w")
        # sensible empty view until a field arrives
        self.set_auto_limits("t", (dl[0] / 2, dl[1] / 2), (0, 1.05))
        self.set_auto_limits("w", (p.w0_rad_fs / 2 + (sl[0] - p.w0_rad_fs) / 2,
                                   p.w0_rad_fs / 2 + (sl[1] - p.w0_rad_fs) / 2), (0, 1.05))
        self._update_artists()
        self.finish_figure()

    def _update_artists(self):
        p = self.prepared
        if p is None or "ret" not in self._art:
            return
        a = self._art
        if self.trace is not None:
            a["ret"].set_data(self.trace)
        if self.G:
            a["g_text"].set_text(f"G  {self.G[-1]:.3e}\nG′ {self.Gp[-1]:.3e}\n"
                                 f"it {self.iters[-1]}")
            a["g_text"].set_visible(True)
        if self.E is not None:
            v = pulse_views(self.E, p.t_fs, p.w_rad_fs, p.w0_rad_fs)
            a["It"].set_ydata(v["It"]); a["phi_t"].set_ydata(v["phi_t"])
            a["Iw"].set_ydata(v["Iw"]); a["phi_w"].set_ydata(v["phi_w"])
            self.set_auto_limits("t", visible_range(v["t"], v["It"], 1e-3, 0.2))
            self.set_auto_limits("w", visible_range(v["w"], v["Iw"], 1e-3, 0.2))
            for key in ("tp", "wp"):
                if not self.is_manual(key):
                    ax = self._axes[key]
                    ax.relim(); ax.autoscale_view(scalex=False)
            a["t_text"].set_text(f"FWHM {v['fwhm_t']:.1f} fs")
            a["w_text"].set_text(f"FWHM {v['fwhm_w']:.3f} rad/fs\n"
                                 f"λ₀ {v['lambda0_nm']:.0f} nm")
