"""Shared matplotlib canvas styling, Lillypad's look (FrogCanvas._style).

Every colour is read from design.tokens.PALETTE at draw time, so a theme
switch is `restyle()` + a redraw. Every tick (major AND minor), offset text
and label is coloured explicitly: anything left to matplotlib comes out
black, which is invisible on the dark plot background.

Decorations are kept small on purpose — every pixel a label takes is one the
traces do not get.
"""

from __future__ import annotations

import tadpole.design.mpl_style  # noqa: F401  (selects the QtAgg backend first)

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, Locator, MaxNLocator, NullLocator
from PySide6.QtCore import Signal

from tadpole.design.mpl_style import PICTURE_DPI, PICTURE_MAX_PX
from tadpole.design.tokens import PALETTE
from tadpole.gui.canvas import layout
from tadpole.gui.canvas.interaction import InteractionMixin

C_NM_PER_FS = 299.792458
TWO_PI_C = 2 * np.pi * C_NM_PER_FS

# Type sizes on the plots (points).
TICK_FS = 7.5
LABEL_FS = 8.5
TITLE_FS = 10
BOX_FS = 7.5



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


class LambdaLocator(Locator):
    """Ticks for a λ axis drawn over an ω axis.

    λ = 2πc/ω, so ticks spaced evenly in λ bunch up at low ω ("6000 4000"
    on top of each other). These are spaced evenly in ω instead and labelled
    with λ rounded to the fewest significant digits that keep them distinct.
    """
    def __init__(self, nbins=5):
        self._omega = MaxNLocator(nbins=nbins)

    def __call__(self):
        lo, hi = sorted(self.axis.get_view_interval())       # in λ
        w_lo, w_hi = TWO_PI_C / hi, TWO_PI_C / max(lo, 1e-6)
        ws = [w for w in self._omega.tick_values(w_lo, w_hi) if w_lo <= w <= w_hi and w > 0]
        lams = TWO_PI_C / np.asarray(ws, float)
        for digits in (2, 3, 4):
            r = np.array([float(f"{x:.{digits}g}") for x in lams])
            if len(set(r)) == len(r):
                return sorted(r)
        return sorted(lams)


def _lambda_format(v, _pos):
    return f"{v:.0f}" if v >= 10 else f"{v:.3g}"


class TadpoleCanvas(InteractionMixin, FigureCanvasQTAgg):
    """Base canvas: a figure in the theme's plot colours on a FIXED grid
    layout (layout.py: the panels never move with their numbers), with
    Lillypad's mouse interaction (interaction.py)."""

    message = Signal(str)      # for the status bar (e.g. a mistyped bound)

    def __init__(self, parent=None):
        self.fig = Figure(dpi=100)
        super().__init__(self.fig)
        self._cells = {}           # Axes → (row, col) of the current grid
        self._shape = (1, 1)
        self._ratios = (None, None)
        self._cbar_slots = []      # (Colorbar, parent axes)
        self.mpl_connect("resize_event", lambda _e: self.apply_layout())
        self.setParent(parent)
        self.cmap_name = "jet"
        self.cmap_reversed = False
        self.log_scale = False
        self._init_interaction()

    # ── figure lifecycle ─────────────────────────────────────────────────
    def begin_figure(self):
        self.fig.clear()
        self.fig.set_facecolor(PALETTE["plot_bg"])
        self._reset_registry()
        self._cells, self._cbar_slots = {}, []

    def grid(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """The figure's panels, as an nrows×ncols array, placed by
        layout.place rather than by matplotlib."""
        import numpy as _np
        self._shape = (nrows, ncols)
        self._ratios = (height_ratios, width_ratios)
        axs = _np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axs[r, c] = self.fig.add_axes([0, 0, 1, 1])
                self._cells[axs[r, c]] = (r, c)
        return axs

    def apply_layout(self):
        if self._cells:
            layout.place(self.fig, self._cells, self._shape, *self._ratios,
                         cbars=self._cbar_slots)

    def finish_figure(self):
        self.apply_layout()
        self.finish_views()
        self.draw_idle()

    # ── styling ──────────────────────────────────────────────────────────
    @staticmethod
    def _ticks(ax, color=None):
        c = color or PALETTE["text_dim"]
        ax.tick_params(which="both", colors=c, labelcolor=c, labelsize=TICK_FS)
        ax.tick_params(which="major", length=3, pad=1.5)
        ax.tick_params(which="minor", length=1.8, pad=1.5)
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_color(c)
            axis.get_offset_text().set_fontsize(TICK_FS)

    def style_axes(self, ax, title="", xlabel="", ylabel="", grid=True):
        ax.set_facecolor(PALETTE["plot_bg"])
        self._ticks(ax)
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["border"])
        if xlabel:
            ax.set_xlabel(xlabel, color=PALETTE["text_dim"], fontsize=LABEL_FS, labelpad=1)
        if ylabel:
            ax.set_ylabel(ylabel, color=PALETTE["text_dim"], fontsize=LABEL_FS, labelpad=1)
        if title:
            self.title(ax, title)
        if grid:
            ax.grid(True, color=PALETTE["grid"], lw=0.6, ls="--", alpha=0.7)
        else:
            ax.grid(False)

    def title(self, ax, text):
        ax.set_title(text, color=PALETTE["accent"], fontsize=TITLE_FS,
                     fontweight="bold", loc="left", pad=3)

    def style_twin(self, ax, ylabel, color):
        self._ticks(ax)
        ax.tick_params(axis="y", which="both", colors=color, labelcolor=color)
        for s in ax.spines.values():
            s.set_edgecolor(PALETTE["border"])
        ax.set_ylabel(ylabel, color=color, fontsize=LABEL_FS, labelpad=1)

    def colorbar(self, mappable, ax, label=""):
        """A colour bar to the right of `ax` (an Axes, or a list of them in
        one row sharing the bar); layout.place positions it."""
        parents = list(ax) if isinstance(ax, (list, tuple)) else [ax]
        cb = self.fig.colorbar(mappable, cax=self.fig.add_axes([0, 0, 0.01, 0.01]))
        self._cbar_slots.append((cb, parents))
        cb.outline.set_edgecolor(PALETTE["border"])
        self._ticks(cb.ax)
        if label:
            cb.set_label(label, color=PALETTE["text_dim"], fontsize=TICK_FS, labelpad=1)
        return cb

    def lambda_secondary(self, ax, which="x"):
        """λ (nm) on the other side of an ω axis, ticks evenly spaced in ω.
        The unit sits in the first tick label instead of an axis label, which
        would cost a whole row."""
        loc = "top" if which == "x" else "right"
        make = ax.secondary_xaxis if which == "x" else ax.secondary_yaxis
        sec = make(loc, functions=(lambda_axis, lambda_axis))
        axis = sec.xaxis if which == "x" else sec.yaxis
        axis.set_major_locator(LambdaLocator())
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(_lambda_format))
        self._ticks(sec)
        for s in sec.spines.values():
            s.set_edgecolor(PALETTE["border"])
        if which == "x":
            sec.set_xlabel("λ (nm)", color=PALETTE["text_dim"], fontsize=TICK_FS,
                           labelpad=1, loc="right")
        else:
            sec.set_ylabel("λ (nm)", color=PALETTE["text_dim"], fontsize=TICK_FS,
                           labelpad=1, loc="top")
        return sec

    def legend(self, ax, **kw):
        leg = ax.legend(fontsize=BOX_FS, frameon=True, borderpad=0.3,
                        handlelength=1.5, **kw)
        leg.get_frame().set_facecolor(PALETTE["surface"])
        leg.get_frame().set_edgecolor(PALETTE["border"])
        for t in leg.get_texts():
            t.set_color(PALETTE["text"])
        return leg

    def info_box(self, ax, text, loc=(0.02, 0.97), fontsize=BOX_FS):
        return ax.text(*loc, text, transform=ax.transAxes, va="top", ha="left",
                       fontsize=fontsize, color=PALETTE["text"], family="monospace",
                       zorder=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["surface"],
                                 edgecolor=PALETTE["border"], alpha=0.85))

    def empty_message(self, ax, text):
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                color=PALETTE["text_dim"], fontsize=11)

    # ── appearance settings ──────────────────────────────────────────────
    def set_appearance(self, cmap=None, reversed_=None, log=None):
        if cmap is not None:
            self.cmap_name = cmap
        if reversed_ is not None:
            self.cmap_reversed = reversed_
        if log is not None and log != self.log_scale:
            self.log_scale = log
            # a colour range typed for one scale is meaningless on the other
            for key in [k for k in self._manual if k in self._cbars]:
                self._manual.pop(key)
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
