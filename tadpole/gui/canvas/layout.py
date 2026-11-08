"""A fixed grid layout for the canvases (in the spirit of Lillypad's layout.py).

matplotlib's constrained layout re-fits every panel to its current tick
labels on every draw, so the plots jumped whenever a number grew a digit or
lost a minus sign. Here each panel's margins come from its STRUCTURE — does
it have a title, tick labels, an axis label, a twin axis, a λ axis, a colour
bar — never from the text in them, so the axes only move when the window is
resized.

Budgets are in points (they hold text, which is sized in points) and turned
into pixels with the figure's dpi, so they follow the display scaling. Panels
are placed on an aligned grid: every column gets the widest left/right
margin any of its rows needs, every row the tallest top/bottom margin, and
the space left over is shared by the axes in proportion to the ratios.
"""

from __future__ import annotations

from matplotlib.axes import Axes

# Budgets, in points. Tick labels are 7.5 pt, axis labels 8.5 pt, titles 10 pt.
OUTER = 3          # figure edge
GAP = 5            # between neighbouring panels
TICK_W = 22        # a column of y tick labels (up to ~5 characters) + ticks
LOG_TICK_W = 30    # …log-axis labels like 4×10⁻²
TICK_H = 12        # a row of x tick labels + ticks
LABEL = 10         # one line of axis label
TITLE = 14         # a panel title + its pad
TITLE_PAD = 3
CBAR_GAP = 4
CBAR_W = 7
CBAR_TICKS = 25    # colour-bar tick labels (up to "−0.015")


def _labels_on(ax: Axes, side: str) -> bool:
    axis = ax.xaxis if side in ("bottom", "top") else ax.yaxis
    if not axis.get_visible():
        return False
    default = side in ("bottom", "left")
    return bool(axis.get_tick_params(which="major").get(f"label{side}", default))


def _twins(ax: Axes):
    try:
        sib = ax._twinned_axes.get_siblings(ax)
    except AttributeError:                       # pragma: no cover
        return []
    return [a for a in sib if a is not ax]


def _secondaries(ax: Axes, orientation: str):
    return [c for c in getattr(ax, "child_axes", [])
            if getattr(c, "_orientation", None) == orientation]


def _y_width(ax: Axes, side: str) -> float:
    if not _labels_on(ax, side):
        return 0.0
    return LOG_TICK_W if ax.get_yscale() == "log" else TICK_W


def margins(ax: Axes) -> dict:
    """What `ax` needs around it, in points — from structure only."""
    left = GAP + _y_width(ax, "left") + (LABEL if ax.get_ylabel() else 0)
    right = GAP
    for t in _twins(ax):
        right += _y_width(t, "right") + (LABEL if t.get_ylabel() else 0)
    for s in _secondaries(ax, "y"):
        right += TICK_W + (LABEL if s.get_ylabel() else 0)
    top = (TITLE if (ax.get_title(loc="left") or ax.get_title()) else GAP)
    for s in _secondaries(ax, "x"):
        top += TICK_H + (LABEL if s.get_xlabel() else 0)
    bottom = GAP + (TICK_H if _labels_on(ax, "bottom") else 0) \
        + (LABEL if ax.get_xlabel() else 0)
    return dict(left=left, right=right, top=top, bottom=bottom)


def cbar_width(cbar) -> float:
    return CBAR_GAP + CBAR_W + CBAR_TICKS + (LABEL if cbar.ax.get_ylabel() else 0)


def place(fig, cells, shape, height_ratios=None, width_ratios=None, cbars=()):
    """Position the grid. `cells`: {Axes: (row, col)}; `cbars`: (colorbar,
    parent axes list) — each bar sits to the right of its rightmost parent,
    after that parent's own right-hand decorations."""
    nr, nc = shape
    hr = list(height_ratios or [1] * nr)
    wr = list(width_ratios or [1] * nc)
    px = fig.dpi / 72.0
    W, H = fig.bbox.width, fig.bbox.height
    if W < 10 or H < 10:
        return
    m = {ax: margins(ax) for ax in cells}
    L, R = [0.0] * nc, [0.0] * nc
    T, B = [0.0] * nr, [0.0] * nr
    for ax, (r, c) in cells.items():
        L[c] = max(L[c], m[ax]["left"])
        R[c] = max(R[c], m[ax]["right"])
        T[r] = max(T[r], m[ax]["top"])
        B[r] = max(B[r], m[ax]["bottom"])
    extra = {}                                   # Axes → colour-bar room
    for cb, parents in cbars:
        host = max(parents, key=lambda a: cells.get(a, (0, -1))[1])
        if host in cells:
            r, c = cells[host]
            R[c] = max(R[c], m[host]["right"] + cbar_width(cb))
            extra[cb] = host
    free_w = W - 2 * OUTER * px - sum(L + R) * px
    free_h = H - 2 * OUTER * px - sum(T + B) * px
    aw = [max(free_w, 10 * nc) * w / sum(wr) for w in wr]
    ah = [max(free_h, 10 * nr) * h / sum(hr) for h in hr]
    xs, x = [], OUTER * px
    for c in range(nc):
        x += L[c] * px
        xs.append(x)
        x += aw[c] + R[c] * px
    ys, y = [], H - OUTER * px                  # rows top-down; mpl y is bottom-up
    for r in range(nr):
        y -= T[r] * px + ah[r]
        ys.append(y)
        y -= B[r] * px
    for ax, (r, c) in cells.items():
        ax.set_position([xs[c] / W, ys[r] / H, aw[c] / W, ah[r] / H])
        # Titles of one row on one line: each sits at the top of the row's
        # margin, above whatever (a λ axis) the tallest neighbour has there.
        dy = (T[r] - TITLE + TITLE_PAD) * px / ah[r]
        ax._autotitlepos = False
        for t in (ax.title, ax._left_title, ax._right_title):
            t.set_y(1 + dy)
    for cb, parents in cbars:
        host = extra.get(cb)
        if host is None:
            continue
        x0 = (xs[cells[host][1]] + aw[cells[host][1]]
              + (m[host]["right"] - GAP + CBAR_GAP) * px)
        rows = [cells[p][0] for p in parents if p in cells]
        y0 = min(ys[r] for r in rows)
        y1 = max(ys[r] + ah[r] for r in rows)
        cb.ax.set_position([x0 / W, y0 / H, CBAR_W * px / W, (y1 - y0) / H])
