"""The Reconstruct page: drives the Julia engine and shows its progress.

The engine lives in another process; a 60 ms QTimer polls it on the GUI
thread (Lillypad's display-timer pattern), so there is no QThread here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from tadpole.analysis import pulse_views
from tadpole.engine.client import EngineClient
from tadpole.engine.protocol import NOT_IMPLEMENTED, Done, Error, Log, Snapshot
from tadpole.gui.window.build_ui import ALGORITHM_NOTES
from tadpole.io.prepared import PulseSource, load_prepared, save_result

POLL_MS = 60

ENGINE_LAMP = {"off": "unknown", "starting": "warn", "ready": "ok",
               "busy": "busy", "error": "sat"}
ENGINE_TEXT = {"off": "Julia: off", "starting": "Julia: starting…",
               "ready": "Julia: ready", "busy": "Julia: running", "error": "Julia: error"}


class ReconController:
    def __init__(self, win, engine: EngineClient | None = None):
        self.win = win
        self.engine = engine or EngineClient()
        self._t_start = 0.0
        self._max_iter = 1
        self._timer = QTimer(win)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.poll)

        win.btn_start.clicked.connect(self.start)
        win.btn_stop.clicked.connect(self.stop)
        win.btn_load_prep.clicked.connect(self.load_dialog)
        win.btn_save_result.clicked.connect(self.save_dialog)
        win.btn_save_analysis.clicked.connect(self.save_dialog)
        win.btn_recon_adv.clicked.connect(
            lambda: win.dlg_recon_adv.show_near(win.btn_recon_adv))
        win.cmb_alg.currentIndexChanged.connect(self._algorithm_changed)
        self._algorithm_changed()
        self._last_state = None
        self._update_engine_status()

    def start_engine(self):
        if os.environ.get("TADPOLE_NO_ENGINE"):
            # GUI-only mode: Prepare works, Reconstruct cannot start.
            self.win.lbl_engine.setText("Julia: disabled (TADPOLE_NO_ENGINE)")
            return
        self.engine.start()
        self._timer.start()
        self._update_engine_status()

    def shutdown(self):
        self._timer.stop()
        self.engine.shutdown()

    # ── input ────────────────────────────────────────────────────────────
    def set_input(self, prepared):
        w = self.win
        w.state.recon_input = prepared
        w.state.result = None
        src = Path(prepared.source).name if prepared.source not in ("", "simulated") \
            else (prepared.source or "prepared trace")
        w.lbl_recon_input.setText(src)
        w.lbl_recon_input_info.setText(
            f"{prepared.n}×{prepared.n} grid, dt = {prepared.dt_fs:.3f} fs\n"
            f"signal λ₀ = {prepared.lambda0_nm:.1f} nm, "
            f"{prepared.delays_meas_fs.size} measured delays")
        w.recon_canvas.set_prepared(prepared)
        w.lbl_result.setText("No result yet")
        self._clear_analysis()
        self._update_buttons()

    def load_dialog(self):
        start = self.win.settings_ctl.last_dir()
        path, _ = QFileDialog.getOpenFileName(self.win, "Load prepared trace", start,
                                              "Prepared trace (*.npz)")
        if not path:
            return
        try:
            self.set_input(load_prepared(path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self.win, "Could not load", str(exc))

    # ── options ──────────────────────────────────────────────────────────
    def algorithm(self) -> str:
        return self.win.cmb_alg.currentData()

    def options(self) -> dict:
        w = self.win
        return {
            "max_iter": w.sp_max_iter.value(),
            "tol": 10.0 ** w.sp_tol_exp.value(),
            "init": w.cmb_init.currentData(),
            "seed": w.sp_seed.value(),
            "alpha_min": w.sp_alpha_min.value(),
            "alpha_max": w.sp_alpha_max.value(),
            "gamma": w.sp_gamma.value(),
            "snapshot_every": 1,
            "min_snapshot_s": 0.1,
        }

    def _algorithm_changed(self, *_):
        alg = self.algorithm()
        self.win.cmb_alg.setToolTip(ALGORITHM_NOTES.get(alg, ""))
        for lbl, wdg in self.win.ptycho_rows:
            lbl.setVisible(alg == "ptychographic")
            wdg.setVisible(alg == "ptychographic")
        self._update_buttons()

    # ── run ──────────────────────────────────────────────────────────────
    def start(self):
        w = self.win
        p = w.state.recon_input
        if p is None or self.engine.state != "ready":
            return
        alg = self.algorithm()
        opts = self.options()
        self._run_opts = dict(opts)          # shown with the result on Analysis
        self._max_iter = max(1, opts["max_iter"])
        if alg == "ptychographic":
            self.engine.submit(alg, p.trace_meas, p.dt_fs, mask=p.mask_meas,
                               delays_fs=p.delays_meas_fs, options=opts)
        else:
            self.engine.submit(alg, p.trace, p.dt_fs, mask=p.mask, options=opts)
        self._t_start = time.perf_counter()
        w.recon_canvas.reset_progress()
        w.progress.setValue(0)
        w.lbl_progress.setText("starting…")
        w.lbl_result.setObjectName("dim")
        w.lbl_result.setText(f"Running {alg.upper()}…")
        w.lbl_result.style().polish(w.lbl_result)
        self._update_buttons()

    def stop(self):
        self.engine.stop()

    def poll(self):
        for msg in self.engine.poll():
            if isinstance(msg, Snapshot):
                self._on_snapshot(msg)
            elif isinstance(msg, Done):
                self._on_done(msg)
            elif isinstance(msg, Error):
                self._on_error(msg)
            elif isinstance(msg, Log):
                self.win.statusBar().showMessage(msg.text, 5000)
        self._update_engine_status()

    def _on_snapshot(self, s: Snapshot):
        w = self.win
        w.progress.setValue(int(100 * min(1.0, s.iteration / self._max_iter)))
        w.lbl_progress.setText(f"iter {s.iteration} / {self._max_iter}  ·  "
                             f"{time.perf_counter() - self._t_start:.1f} s")
        w.recon_canvas.show_progress(s.iteration, s.G, s.Gp, s.trace, s.E)

    def _on_done(self, d: Done):
        w = self.win
        r = d.result
        w.state.result = r
        w.recon_canvas.show_result(r)
        w.progress.setValue(100)
        w.lbl_progress.setText(f"done: {r['iterations']} iterations  ·  "
                             f"{r['elapsed_s']:.1f} s")
        why = {"tolerance": "reached tolerance", "max_iter": "hit max iterations",
               "user": "stopped"}.get(r["stopped"], r["stopped"])
        w.lbl_result.setObjectName("dim")
        w.lbl_result.setText(f"{r['algorithm'].upper()}: {why} after "
                             f"{r['iterations']} iterations. Best G = {r['G']:.3e}, "
                             f"G′ = {r['Gp']:.3e}.")
        w.lbl_result.style().polish(w.lbl_result)
        self._show_analysis(r, why)
        self._update_buttons()

    # ── analysis page ────────────────────────────────────────────────────
    def _show_analysis(self, r, why):
        w = self.win
        p = w.state.recon_input
        w.analysis_canvas.set_result(p, r)
        lines = [f"{r['algorithm'].upper()}, {why}",
                 f"{r['iterations']} iterations, {r['elapsed_s']:.1f} s",
                 f"G  = {r['G']:.3e}", f"G′ = {r['Gp']:.3e}"]
        errs = np.asarray(r.get("errors", []), float)
        if errs.size:
            lines.append(f"best G at iteration {int(np.argmin(errs)) + 1}")
        try:
            v = pulse_views(r["E"], p.t_fs, p.w_rad_fs, p.w0_rad_fs)
            lines += [f"FWHM {v['fwhm_t']:.1f} fs (TL {v['fwhm_tl']:.1f})",
                      f"FWHM {v['fwhm_lam']:.1f} nm, λ₀ {v['lambda0_nm']:.0f} nm"]
        except (KeyError, ValueError, AttributeError):
            pass
        o = getattr(self, "_run_opts", None)
        if o:
            init = o.get("init", "")
            seed = f"seed {o.get('seed', 0)}" + ("" if init == "random" else " (unused)")
            lines += ["", f"init  {init}", seed,
                      f"max {o.get('max_iter', '')} · tol {o.get('tol', 0):.0e}"]
        w.lbl_analysis.setText("\n".join(lines))
        w.btn_mode_analysis.setEnabled(True)
        w.propagator.set_pulse(PulseSource.from_result(p, r))

    def _clear_analysis(self):
        w = self.win
        w.analysis_canvas.clear()
        w.lbl_analysis.setText("No result yet")
        w.btn_mode_analysis.setEnabled(False)
        if w.stack.currentIndex() == 2:
            w.set_mode(1)

    def _on_error(self, e: Error):
        w = self.win
        w.lbl_result.setObjectName("err")
        w.lbl_result.setText(e.message)
        w.lbl_result.style().polish(w.lbl_result)
        w.statusBar().showMessage(e.message.splitlines()[0] if e.message else "Error",
                                  8000)
        w.progress.setValue(0)
        w.lbl_progress.setText("error")
        self._update_buttons()

    # ── status ───────────────────────────────────────────────────────────
    def _update_engine_status(self):
        st = self.engine.state
        if st == self._last_state:
            return
        self._last_state = st
        w = self.win
        w.engine_lamp.set_state(ENGINE_LAMP.get(st, "unknown"))
        text = ENGINE_TEXT.get(st, st)
        if st in ("ready", "busy") and self.engine.julia_version:
            text += f" ({self.engine.julia_version})"
        w.lbl_engine.setText(text)
        w.lbl_engine.setToolTip(self.engine.last_error if st == "error" else "")
        self._update_buttons()

    def _update_buttons(self):
        w = self.win
        st = self.engine.state
        has_input = w.state.recon_input is not None
        implemented = self.algorithm() not in NOT_IMPLEMENTED
        w.btn_start.setEnabled(has_input and st == "ready" and implemented)
        if not implemented:
            w.btn_start.setToolTip("This algorithm is not implemented yet")
        elif st != "ready":
            w.btn_start.setToolTip(ENGINE_TEXT.get(st, st))
        else:
            w.btn_start.setToolTip("")
        w.btn_stop.setEnabled(st == "busy")
        w.btn_save_result.setEnabled(w.state.result is not None)
        w.btn_save_analysis.setEnabled(w.state.result is not None)
        w.cmb_alg.setEnabled(st != "busy")

    # ── saving ───────────────────────────────────────────────────────────
    def save_dialog(self):
        w = self.win
        if w.state.result is None or w.state.recon_input is None:
            return
        start = str(Path(w.settings_ctl.last_dir()) / "retrieval.npz")
        path, _ = QFileDialog.getSaveFileName(
            w, "Save retrieval result", start,
            "Full result (*.npz);;Field as text (*.csv)")
        if path:
            save_result(path, w.state.recon_input, w.state.result)
            w.statusBar().showMessage(f"Saved {path}", 5000)
