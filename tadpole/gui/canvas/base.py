"""Shared matplotlib canvas styling, Lillypad's look (FrogCanvas._style).

Every colour is read from design.tokens.PALETTE at draw time, so a theme
switch is `restyle()` + a redraw.
"""

from __future__ import annotations

import tadpole.design.mpl_style  # noqa: F401  (selects the QtAgg backend first)

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from tadpole.design.mpl_style import PICTURE_DPI, PICTURE_MAX_PX
from tadpole.design.tokens import PALETTE

C_NM_PER_FS = 299.792458
TWO_PI_C = 2 * np.pi * C_NM_PER_FS


def trace_cmap(name: str, reversed_: bool = False):
    cm = matplotlib.colormaps[name]
    if reversed_:
        cm = cm.reversed()
    return cm.with_extremes(bad=PALETTE["plot_bg"])


def trace_norm(log: bool, vmax: float = 1.0):
    vmax = vmax if vmax > 0 else 1.0
    return LogNorm(vmin=vmax * 1e-3, vmax=vmax, clip=True) if log else Normalize(0, vmax)


def lambda_axis(value):
    """ω [rad/fs] ↔ λ [nm] — the same function both ways."""
    # Clamped so an axis whose limits touch ω ≤ 0 (an empty panel's default
    # 0…1) maps to a huge but FINITE λ: matplotlib rejects inf/nan limits.
    v = np.maximum(np.asarray(value, float), 1e-6)
    return TWO_PI_C / v


class TadpoleCanvas(FigureCanvasQTAgg):
    """Base canvas: a constrained-layout figure in the theme's plot colours."""

    def __init__(self, parent=None):
        self.fig = Figure(dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.cmap_name = "jet"
        self.cmap_reversed = False
        self.log_scale = False
        self._colorbars = []

    # ── styling ──────────────────────────────────────────────────────────
    def style_axes(self, ax, title="", xlabel="", ylabel="", grid=True):
        ax.set_facecolor(PALETTE["plot_bg"])
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=8.5)
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["border"])
        if xlabel:
            ax.set_xlabel(xlabel, color=PALETTE["text_dim"], fontsize=9.5, labelpad=2)
        if ylabel:
            ax.set_ylabel(ylabel, color=PALETTE["text_dim"], fontsize=9.5, labelpad=2)
        if title:
            ax.set_title(title, color=PALETTE["accent"], fontsize=11,
                         fontweight="bold", loc="left")
        if grid:
            ax.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        else:
            ax.grid(False)

    def style_twin(self, ax, ylabel, color):
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=8.5)
        ax.tick_params(axis="y", colors=color)
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["border"])
        ax.set_ylabel(ylabel, color=color, fontsize=9.5, labelpad=2)

    def style_colorbar(self, cb):
        cb.outline.set_edgecolor(PALETTE["border"])
        cb.ax.tick_params(colors=PALETTE["text_dim"], labelsize=7.5)

    def style_secondary(self, sec, label, which="y"):
        sec.tick_params(colors=PALETTE["text_dim"], labelsize=8)
        if which == "y":
            sec.set_ylabel(label, color=PALETTE["text_dim"], fontsize=9)
        else:
            sec.set_xlabel(label, color=PALETTE["text_dim"], fontsize=9)
        for s in sec.spines.values():
            s.set_edgecolor(PALETTE["border"])

    def legend(self, ax, **kw):
        leg = ax.legend(fontsize=8, frameon=True, **kw)
        leg.get_frame().set_facecolor(PALETTE["surface"])
        leg.get_frame().set_edgecolor(PALETTE["border"])
        for t in leg.get_texts():
            t.set_color(PALETTE["text"])
        return leg

    def info_box(self, ax, text, loc=(0.02, 0.97)):
        return ax.text(*loc, text, transform=ax.transAxes, va="top", ha="left",
                       fontsize=8.5, color=PALETTE["text"], family="monospace",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["surface"],
                                 edgecolor=PALETTE["border"], alpha=0.85))

    def empty_message(self, ax, text):
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                color=PALETTE["text_disabled"], fontsize=11)

    # ── appearance settings ──────────────────────────────────────────────
    def set_appearance(self, cmap=None, reversed_=None, log=None):
        if cmap is not None:
            self.cmap_name = cmap
        if reversed_ is not None:
            self.cmap_reversed = reversed_
        if log is not None:
            self.log_scale = log
        self.redraw()

    def redraw(self):
        """Rebuild the whole figure from current data and PALETTE."""
        raise NotImplementedError

    def restyle(self):
        self.fig.set_facecolor(PALETTE["plot_bg"])
        self.redraw()

    # ── export ───────────────────────────────────────────────────────────
    def save_picture(self, path):
        w, h = self.fig.get_size_inches()
        dpi = min(PICTURE_DPI, PICTURE_MAX_PX / max(w, h))
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
