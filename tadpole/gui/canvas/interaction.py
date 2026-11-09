"""Mouse work on every Tadpole canvas, the Lillypad way
(lillypad/gui/canvas/interaction.py), made generic over registered panels.

  * **left-drag** in a panel: rubber-band zoom;
  * **right-click** in a panel: back to its automatic view;
  * **click an end tick label**: type that bound;
  * on a **colour bar**: drag up/down it to pick the colour range, the wheel
    stretches the top of the range, right-click returns to auto, and its end
    labels can be typed into like any axis. A colour bar can drive several
    images (Measured and Retrieved share one).

Canvases register what can be touched during `redraw()`:

    register_view(key, ax, parent=None, group=None)
    register_cbar(key, cbar, images)

and call `finish_views()` at the end. A zoom or a typed range is MANUAL and
stored per key, so it survives the figure being rebuilt (theme switch,
colormap change, the final result arriving); live updates go through
`set_auto_limits`, which respects it. Right-click forgets it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QRubberBand

from tadpole.gui.widgets import AxisBoundEditor


class InteractionMixin:
    CLICK_PX = 5             # a press that moves less than this is a click
    CBAR_WHEEL_STEP = 1.07   # per wheel notch, factor on the colour span
    AXIS_EDIT_PAD = 3        # px of slack around a tick label's ink

    def _init_interaction(self):
        self._manual = {}    # key -> (xlim, ylim) for a view, (lo, hi) for a colour bar
        self._reset_registry()
        self._drag = None    # (axes, x0, y0) in mpl display px
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._axis_editor = AxisBoundEditor(self)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)

    # ── registry ─────────────────────────────────────────────────────────
    def _reset_registry(self):
        self._views = {}     # key -> dict(ax, home=[xlim, ylim], parent, group)
        self._cbars = {}     # key -> dict(cbar, images, home=(lo, hi))
        self._key_of = {}    # Axes -> key

    def register_view(self, key, ax, parent=None, group=None):
        """`parent`: for a twin axis, the panel a drag on it zooms. `group`:
        views that share axes and so are zoomed and reset together."""
        assert key not in self._cbars, f"{key!r} is already a colour bar"
        self._views[key] = dict(ax=ax, home=[None, None], parent=parent, group=group,
                                explicit=False)
        self._key_of[ax] = key

    def register_cbar(self, key, cbar, images):
        # one namespace with the views: _key_of maps an Axes to either kind
        assert key not in self._views, f"{key!r} is already a panel"
        self._cbars[key] = dict(cbar=cbar, images=list(images),
                                home=tuple(images[0].get_clim()))
        self._key_of[cbar.ax] = key

    def set_auto_limits(self, key, xlim=None, ylim=None):
        """The automatic view of `key`: remembered for right-click, applied
        unless the user has zoomed this panel."""
        v = self._views.get(key)
        if v is None:
            return
        v["explicit"] = True
        if xlim is not None:
            v["home"][0] = tuple(xlim)
        if ylim is not None:
            v["home"][1] = tuple(ylim)
        if key in self._manual:
            return
        if xlim is not None:
            v["ax"].set_xlim(*xlim)
        if ylim is not None:
            v["ax"].set_ylim(*ylim)

    def is_manual(self, key) -> bool:
        return key in self._manual

    def clear_manual(self):
        self._manual.clear()

    def finish_views(self):
        """End of redraw(): record each panel's automatic view, then put back
        whatever the user had zoomed to."""
        for key, v in self._views.items():
            ax = v["ax"]
            if v["home"][0] is None:
                v["home"][0] = tuple(ax.get_xlim())
            if v["home"][1] is None:
                v["home"][1] = tuple(ax.get_ylim())
            m = self._manual.get(key)
            if m is not None:
                ax.set_xlim(*m[0])
                ax.set_ylim(*m[1])
        for key, c in self._cbars.items():
            m = self._manual.get(key)
            if m is not None:
                self._set_clim(c, *m)

    def _members(self, key):
        """`key` plus every view zoomed with it: its group and its twins."""
        v = self._views[key]
        out = {key}
        if v["group"] is not None:
            out |= {k for k, w in self._views.items() if w["group"] == v["group"]}
        out |= {k for k, w in self._views.items() if w["parent"] in out}
        return out

    def _record(self, keys):
        for k in keys:
            ax = self._views[k]["ax"]
            self._manual[k] = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    # ── commands ─────────────────────────────────────────────────────────
    def reset_view(self, key):
        """Right-click: forget the manual view/colour range, back to auto."""
        if key in self._cbars:
            c = self._cbars[key]
            self._manual.pop(key, None)
            self._set_clim(c, *c["home"])
        elif key in self._views:
            key = self._views[key]["parent"] or key
            for k in self._members(key):
                self._manual.pop(k, None)
                v = self._views[k]
                if v["parent"] is not None and not v["explicit"]:
                    v["ax"].relim()          # a twin without its own auto view
                    v["ax"].autoscale_view(scalex=False)
                    continue
                if v["home"][0] is not None:
                    v["ax"].set_xlim(*v["home"][0])
                if v["home"][1] is not None:
                    v["ax"].set_ylim(*v["home"][1])
        self.draw_idle()

    def _apply_zoom(self, ax, xlo, xhi, ylo, yhi):
        key = self._key_of.get(ax)
        if key is None:
            return
        if key in self._cbars:
            self.set_clim(key, ylo, yhi)
            return
        key = self._views[key]["parent"] or key
        target = self._views[key]["ax"]
        target.set_xlim(xlo, xhi)
        target.set_ylim(ylo, yhi)
        members = self._members(key)
        self._record(members)
        self.draw_idle()

    def set_clim(self, key, lo, hi):
        c = self._cbars[key]
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            return
        if isinstance(c["images"][0].norm, LogNorm) and lo <= 0:
            self.message.emit("A log colour scale needs a range above 0.")
            return
        self._manual[key] = (float(lo), float(hi))
        self._set_clim(c, lo, hi)
        self.draw_idle()

    @staticmethod
    def _set_clim(c, lo, hi):
        for im in c["images"]:
            im.set_clim(lo, hi)

    def apply_axis_bound(self, ax, which, end, value):
        """One typed bound (AxisBoundEditor's commit)."""
        key = self._key_of.get(ax)
        if key is None:
            return
        if key in self._cbars:
            lo, hi = self._cbars[key]["images"][0].get_clim()
        else:
            lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        flipped = lo > hi
        lo, hi = min(lo, hi), max(lo, hi)
        lo, hi = (value, hi) if end == "lo" else (lo, value)
        scale = ax.get_xscale() if which == "x" else ax.get_yscale()
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi \
                or (scale == "log" and lo <= 0):
            self.message.emit(f"{value:g} would leave the axis inverted or empty "
                              f"— axis unchanged.")
            return
        if key in self._cbars:
            self.set_clim(key, lo, hi)
            return
        lims = (hi, lo) if flipped else (lo, hi)
        (ax.set_xlim if which == "x" else ax.set_ylim)(*lims)
        # A twin's own y is its own; anything else drags its group along.
        self._record({key} if self._views[key]["parent"] else self._members(key))
        self.draw_idle()

    # ── hit tests ────────────────────────────────────────────────────────
    def _disp_to_qt(self, x, y):
        """mpl display px (origin bottom-left) → Qt logical px (top-left)."""
        dpr = self.devicePixelRatioF()
        return QPoint(round(x / dpr), round(self.height() - y / dpr))

    def _editable_axes(self):
        for key, v in self._views.items():
            yield v["ax"], "x"
            yield v["ax"], "y"
        for c in self._cbars.values():
            yield c["cbar"].ax, "y"

    @staticmethod
    def _end_tick_labels(ax, which):
        axis = ax.xaxis if which == "x" else ax.yaxis
        if not axis.get_visible():
            return ()
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        lo, hi = min(lo, hi), max(lo, hi)
        i = 0 if which == "x" else 1
        inside = [t for t in axis.get_ticklabels()
                  if t.get_visible() and t.get_text()
                  and lo <= t.get_position()[i] <= hi]
        inside.sort(key=lambda t: t.get_position()[i])
        return (inside[0], inside[-1]) if len(inside) >= 2 else ()

    def _hit_axis_end(self, event):
        try:
            renderer = self.get_renderer()
        except (AttributeError, RuntimeError):
            return None
        pad = self.AXIS_EDIT_PAD
        for ax, which in self._editable_axes():
            if not ax.get_visible():
                continue
            for end, label in zip(("lo", "hi"), self._end_tick_labels(ax, which)):
                try:
                    bb = label.get_window_extent(renderer)
                except (ValueError, RuntimeError):
                    continue
                if (bb.x0 - pad <= event.x <= bb.x1 + pad
                        and bb.y0 - pad <= event.y <= bb.y1 + pad):
                    rect = QRect(self._disp_to_qt(bb.x0, bb.y1),
                                 self._disp_to_qt(bb.x1, bb.y0)).normalized()
                    return ax, which, end, rect
        return None

    def _open_axis_editor(self, hit):
        ax, which, end, rect = hit
        key = self._key_of.get(ax)
        if key in self._cbars:
            lo, hi = self._cbars[key]["images"][0].get_clim()
        else:
            lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
            lo, hi = min(lo, hi), max(lo, hi)
        self._axis_editor.open_on(ax, which, end, lo if end == "lo" else hi, rect)

    # ── mouse ────────────────────────────────────────────────────────────
    def _cancel_drag(self):
        self._rubber.hide()
        self._drag = None

    def _clamped(self, event):
        ax = self._drag[0]
        bb = ax.get_window_extent()
        x = event.x if event.x is not None else self._drag[1]
        y = event.y if event.y is not None else self._drag[2]
        return min(max(x, bb.x0), bb.x1), min(max(y, bb.y0), bb.y1)

    def _is_cbar(self, ax):
        return self._key_of.get(ax) in self._cbars

    def _on_press(self, event):
        ax = event.inaxes
        key = self._key_of.get(ax) if ax is not None else None
        if event.button == 3:
            if self._drag is not None:
                self._cancel_drag()
            elif key is not None:
                self.reset_view(key)
            return
        if event.button != 1:
            return
        if key is not None:
            self._drag = (ax, event.x, event.y)
        elif ax is None:
            hit = self._hit_axis_end(event)
            if hit is not None:
                self._open_axis_editor(hit)

    def _on_motion(self, event):
        if self._drag is None:
            return
        ax, x0, y0 = self._drag
        cx, cy = self._clamped(event)
        moved = max(abs(cx - x0), abs(cy - y0))
        if self._is_cbar(ax):
            bb = ax.get_window_extent()
            x0, cx, moved = bb.x0, bb.x1, abs(cy - y0)
        if moved < self.CLICK_PX * self.devicePixelRatioF():
            self._rubber.hide()
            return
        self._rubber.setGeometry(
            QRect(self._disp_to_qt(x0, y0), self._disp_to_qt(cx, cy)).normalized())
        self._rubber.show()

    def _on_release(self, event):
        if event.button != 1 or self._drag is None:
            return
        ax, x0, y0 = self._drag
        cx, cy = self._clamped(event)
        self._cancel_drag()
        thr = self.CLICK_PX * self.devicePixelRatioF()
        cbar = self._is_cbar(ax)
        if abs(cy - y0) < thr or (not cbar and abs(cx - x0) < thr):
            return
        # A twin (phase) axis zooms its parent: shared x, intensity y.
        key = self._key_of[ax]
        if not cbar and self._views[key]["parent"] is not None:
            ax = self._views[self._views[key]["parent"]]["ax"]
        inv = ax.transData.inverted()
        (dx0, dy0), (dx1, dy1) = inv.transform([(x0, y0), (cx, cy)])
        xlo, xhi = sorted((dx0, dx1))
        ylo, yhi = sorted((dy0, dy1))
        self._apply_zoom(ax, xlo, xhi, ylo, yhi)

    def _on_scroll(self, event):
        """Wheel over a colour bar: up brightens (the top of the range comes
        down), down darkens. The bottom stays put."""
        key = self._key_of.get(event.inaxes) if event.inaxes is not None else None
        if key not in self._cbars or not event.step:
            return
        lo, hi = self._cbars[key]["images"][0].get_clim()
        f = self.CBAR_WHEEL_STEP ** (-event.step)
        if isinstance(self._cbars[key]["images"][0].norm, LogNorm):
            hi = lo * (hi / lo) ** f
        else:
            hi = lo + (hi - lo) * f
        self.set_clim(key, lo, hi)
