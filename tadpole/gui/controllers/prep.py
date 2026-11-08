"""The Prepare page: load a trace, run the preparation pipeline on every
settings change (debounced), preview it, and hand it to the Reconstructor."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from tadpole.io.dwc import read_dwc
from tadpole.io.prepared import save_prepared
from tadpole.prep.model import PrepSettings
from tadpole.prep.pipeline import background_only, prepare
from tadpole.prep.simulate import simulate_raw

DEBOUNCE_MS = 150


class PrepController:
    def __init__(self, win):
        self.win = win
        self._timer = QTimer(win)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.recompute)

        win.btn_open.clicked.connect(self.open_dialog)
        win.btn_simulate.clicked.connect(self.simulate)
        win.btn_send.clicked.connect(self.send)
        win.btn_save_prep.clicked.connect(self.save_dialog)
        win.btn_prep_adv.clicked.connect(
            lambda: win.dlg_prep_adv.show_near(win.btn_prep_adv))
        for cmb in (win.cmb_bg, win.cmb_zero, win.cmb_n):
            cmb.currentIndexChanged.connect(self.schedule)
        for sp in (win.sp_edge_cols, win.sp_edge_rows, win.sp_wl_min, win.sp_wl_max,
                   win.sp_d_min, win.sp_d_max, win.sp_zero_off, win.sp_dt):
            sp.valueChanged.connect(self.schedule)
        win.chk_sym.toggled.connect(self.schedule)
        self._update_buttons()

    # ── settings ↔ widgets ───────────────────────────────────────────────
    def settings(self) -> PrepSettings:
        w = self.win
        return PrepSettings(
            bg_mode=w.cmb_bg.currentData(), bg_edge_cols=w.sp_edge_cols.value(),
            bg_edge_rows=w.sp_edge_rows.value(),
            wl_min_nm=w.sp_wl_min.optional_value(), wl_max_nm=w.sp_wl_max.optional_value(),
            delay_min_fs=w.sp_d_min.optional_value(),
            delay_max_fs=w.sp_d_max.optional_value(),
            zero_mode=w.cmb_zero.currentData(), zero_offset_fs=w.sp_zero_off.value(),
            symmetrize=w.chk_sym.isChecked(), grid_n=int(w.cmb_n.currentData()),
            grid_dt_fs=w.sp_dt.value())

    # ── loading ──────────────────────────────────────────────────────────
    def open_dialog(self):
        start = self.win.settings_ctl.last_dir()
        path, _ = QFileDialog.getOpenFileName(self.win, "Open FROG trace", start,
                                              "FROG trace (*.dwc);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            raw = read_dwc(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self.win, "Could not open trace", str(exc))
            return
        self.win.settings_ctl.set_last_dir(str(Path(path).parent))
        self.load_raw(raw)

    def simulate(self):
        self.load_raw(simulate_raw())

    def load_raw(self, raw):
        w = self.win
        w.state.raw = raw
        name = Path(raw.source).name if raw.source != "simulated" else "Simulated trace"
        w.lbl_file.setText(name)
        wl = raw.wavelengths_nm
        w.lbl_file_info.setText(
            f"{raw.trace.shape[0]} λ × {raw.trace.shape[1]} τ · "
            f"{wl.min():.0f}–{wl.max():.0f} nm · Δτ {raw.delay_step_fs:.3g} fs")
        w.set_mode(0)
        self.recompute()

    # ── pipeline ─────────────────────────────────────────────────────────
    def schedule(self, *_):
        if self.win.state.raw is not None:
            self._timer.start()

    def recompute(self):
        w = self.win
        st = w.state
        if st.raw is None:
            return
        s = self.settings()
        try:
            st.processed = background_only(st.raw, s)
            st.prepared, st.report = prepare(st.raw, s)
        except ValueError as exc:
            st.prepared, st.report = None, None
            w.lbl_grid.setText("")
            self._warn([str(exc)])
        else:
            p, r = st.prepared, st.report
            w.lbl_grid.setText(
                f"dt {p.dt_fs:.3f} fs · dω {p.dw_rad_fs:.4f}\n"
                f"window {r.delay_window_fs:.0f} fs · λ₀ {p.lambda0_nm:.1f} nm")
            self._warn(r.warnings)
        w.prep_canvas.set_data(st.raw, st.processed, st.prepared, st.report)
        self._update_buttons()

    def _warn(self, messages):
        """The first warning in the panel, all of them in its tooltip: the
        column has a fixed height, and several wrapped warnings would not fit."""
        lbl = self.win.lbl_warn
        messages = list(messages)
        if not messages:
            lbl.setText(""); lbl.setToolTip(""); lbl.setVisible(False)
            return
        more = f"  (+{len(messages) - 1} more)" if len(messages) > 1 else ""
        lbl.setText("⚠ " + messages[0] + more)
        lbl.setToolTip("\n".join("⚠ " + m for m in messages))
        lbl.setVisible(True)

    def _update_buttons(self):
        ok = self.win.state.prepared is not None
        self.win.btn_send.setEnabled(ok)
        self.win.btn_save_prep.setEnabled(ok)

    # ── output ───────────────────────────────────────────────────────────
    def send(self):
        p = self.win.state.prepared
        if p is None:
            return
        self.win.recon.set_input(p)
        self.win.set_mode(1)

    def save_dialog(self):
        p = self.win.state.prepared
        if p is None:
            return
        start = str(Path(self.win.settings_ctl.last_dir()) / "prepared.npz")
        path, _ = QFileDialog.getSaveFileName(self.win, "Save prepared trace", start,
                                              "Prepared trace (*.npz)")
        if path:
            save_prepared(path, p)
            self.win.statusBar().showMessage(f"Saved {path}", 5000)
