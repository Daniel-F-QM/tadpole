"""settings.json: what is remembered between sessions (Lillypad's pattern).

One table, `_setting_specs()`, read in both directions as (key, getter,
setter). Adding a setting is one line there — save, the autosave diff and the
restore all walk that list. Polled from the table rather than wired to every
valueChanged signal, because a spinbox drag is not worth a disk write per step.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer

from tadpole.settings import SETTINGS_VERSION, as_bool, load_settings, save_settings

AUTOSAVE_MS = 10_000


def _combo_setter(cmb):
    def set_(v):
        i = cmb.findData(v)
        if i >= 0:
            cmb.setCurrentIndex(i)
    return set_


class SettingsController:
    def __init__(self, win):
        self.win = win
        self._last_saved = None
        self._timer = QTimer(win)
        self._timer.setInterval(AUTOSAVE_MS)
        self._timer.timeout.connect(self.autosave)

    def _setting_specs(self):
        w = self.win
        num = lambda sp: (sp.value, sp.setValue)        # noqa: E731
        opt = lambda sp: (sp.optional_value, sp.set_optional_value)  # noqa: E731
        combo = lambda c: (c.currentData, _combo_setter(c))          # noqa: E731
        chk = lambda c: (c.isChecked, lambda v: c.setChecked(as_bool(v)))  # noqa: E731
        return [
            ("theme", lambda: w.appearance.theme, None),        # restored at startup
            ("window", self._window_geometry, None),             # see show_restored
            ("last_dir", lambda: w.state_last_dir, self.set_last_dir),
            ("cmap", lambda: w.prep_canvas.cmap_name, None),
            ("cmap_reversed", lambda: w.prep_canvas.cmap_reversed, None),
            ("log_scale", lambda: w.prep_canvas.log_scale, None),
            ("bg_mode", *combo(w.cmb_bg)),
            ("bg_edge_cols", *num(w.sp_edge_cols)),
            ("bg_edge_rows", *num(w.sp_edge_rows)),
            ("wl_min_nm", *opt(w.sp_wl_min)),
            ("wl_max_nm", *opt(w.sp_wl_max)),
            ("zero_mode", *combo(w.cmb_zero)),
            ("symmetrize", *chk(w.chk_sym)),
            ("grid_n", *combo(w.cmb_n)),
            ("grid_dt_fs", *num(w.sp_dt)),
            ("algorithm", *combo(w.cmb_alg)),
            ("max_iter", *num(w.sp_max_iter)),
            ("tol_exp", *num(w.sp_tol_exp)),
            ("init", *combo(w.cmb_init)),
            ("seed", *num(w.sp_seed)),
            ("alpha_min", *num(w.sp_alpha_min)),
            ("alpha_max", *num(w.sp_alpha_max)),
            ("gamma", *num(w.sp_gamma)),
        ]

    # ── last directory (file dialogs) ────────────────────────────────────
    def last_dir(self) -> str:
        return self.win.state_last_dir or ""

    def set_last_dir(self, d):
        self.win.state_last_dir = str(d or "")

    def _window_geometry(self):
        w = self.win
        g = w.normalGeometry()
        return {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height(),
                "maximized": w.isMaximized()}

    # ── restore / save ───────────────────────────────────────────────────
    def restore(self):
        data = load_settings()
        for key, _get, set_ in self._setting_specs():
            if set_ is None or key not in data:
                continue
            try:
                set_(data[key])
            except (TypeError, ValueError):
                pass                    # one bad key falls back to its default
        w = self.win
        w.appearance.apply_graphics(data.get("cmap", "jet"),
                                    bool(data.get("cmap_reversed", False)),
                                    bool(data.get("log_scale", False)))
        self._last_saved = self.snapshot()
        self._timer.start()

    def snapshot(self) -> dict:
        out = {"version": SETTINGS_VERSION}
        for key, get, _set in self._setting_specs():
            try:
                out[key] = get()
            except (AttributeError, RuntimeError):
                pass
        return out

    def save(self):
        snap = self.snapshot()
        if save_settings(snap):
            self._last_saved = snap

    def autosave(self):
        snap = self.snapshot()
        if snap != self._last_saved and save_settings(snap):
            self._last_saved = snap
