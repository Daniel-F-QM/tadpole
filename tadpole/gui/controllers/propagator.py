"""The Propagator page: put the retrieved pulse through a stack of material
and compare the result with a trace measured after it.

The material catalogue is read on a worker thread at startup (importing the
refractiveindex package costs ~0.7 s); until it is ready the editor is
disabled. Everything else is computed on a 150 ms debounce in the GUI thread,
except the direction scan (one SHG trace per point), which runs on the same
worker and is polled — a scan whose inputs have since changed is dropped.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCompleter, QFileDialog, QMessageBox

from tadpole.analysis import pulse_views, t2w
from tadpole.gui.canvas.propagator_canvas import PropView
from tadpole.io.dwc import read_dwc
from tadpole.io.prepared import PulseSource, load_prepared, load_result
from tadpole.prep.pipeline import prepare, regrid
from tadpole.propagation import materials as M
from tadpole.propagation.physics import (MODES, best_s, propagate,
                                         pulse_dispersion, scan, shg_trace,
                                         stack_phase, time_reverse,
                                         trace_errors, verdict)

DEBOUNCE_MS = 150
POLL_MS = 50
COVERAGE_WARN = 1e-3      # spectral energy outside a page's data that warns


def _item_text(i, e) -> str:
    name = e.page if e.shelf == "specs" else e.book    # a glass catalogue: the glass
    return f"{i + 1}  {name} · {e.thickness_mm:g} mm"


class PropagatorController:
    def __init__(self, win):
        self.win = win
        self.pulse: PulseSource | None = None
        self.stack: list[M.Element] = [M.Element()]
        self._meas_src = None        # ("dwc", RawTrace) | ("npz", PreparedTrace)
        self.meas = None             # PreparedTrace on the pulse grid
        self.view: PropView | None = None
        self._phase = None           # last applied phase (for the scan)
        self._ready = False
        self._scan_future = None
        self._scan_token = 0
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="propagator")
        self._load_future = self._pool.submit(M.preload)

        self._debounce = QTimer(win)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.recompute)
        self._poll = QTimer(win)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._poll_futures)
        self._poll.start()

        w = win
        for key, name in MODES.items():
            w.cmb_disp_mode.addItem(name, key)
        w.cmb_material.setEnabled(False)
        w.cmb_page.setEnabled(False)
        w.cmb_material.setPlaceholderText("loading materials…")
        w.btn_prop_load_result.clicked.connect(self.load_result_dialog)
        w.btn_prop_load_meas.clicked.connect(self.load_measured_dialog)
        w.btn_stack_add.clicked.connect(self.add_element)
        w.btn_stack_remove.clicked.connect(self.remove_element)
        w.btn_prop_adv.clicked.connect(lambda: w.dlg_prop_adv.show_near(w.btn_prop_adv))
        w.list_stack.currentRowChanged.connect(self._row_selected)
        w.cmb_material.activated.connect(self._material_chosen)
        w.cmb_page.currentIndexChanged.connect(self._page_chosen)
        w.sp_thickness.valueChanged.connect(self._thickness_changed)
        w.cmb_disp_mode.currentIndexChanged.connect(self.schedule)
        w.sp_scan_range.valueChanged.connect(self.schedule)
        w.sp_scan_points.valueChanged.connect(self.schedule)
        self._refresh_list(select=0)
        self._update_buttons()

    def shutdown(self):
        self._poll.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── settings (persistence.py) ────────────────────────────────────────
    def stack_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.stack]

    def set_stack(self, items):
        try:
            stack = [M.Element.from_dict(d) for d in items]
        except (KeyError, TypeError, ValueError):
            return
        self.stack = stack
        self._refresh_list(select=0 if stack else -1)
        self.schedule()

    # ── catalogue (worker thread) ────────────────────────────────────────
    def wait_ready(self, timeout=60):
        """Block until the catalogue is loaded (tests, and nothing else)."""
        self._load_future.result(timeout)
        self._poll_futures()

    def _on_ready(self):
        w = self.win
        try:
            self._load_future.result()
        except Exception as exc:          # broken install: say so, keep going
            w.lbl_element_info.setText(f"Material database unavailable: {exc}")
            return
        self._ready = True
        cmb = w.cmb_material
        cmb.blockSignals(True)
        for shelf, book, label in M.books():
            cmb.addItem(label, f"{shelf}/{book}")
        cmb.setCurrentIndex(-1)
        cmb.blockSignals(False)
        comp = QCompleter(cmb.model(), cmb)
        comp.setFilterMode(Qt.MatchContains)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        comp.activated[str].connect(self._material_typed)
        cmb.setCompleter(comp)
        cmb.setEnabled(True)
        w.cmb_page.setEnabled(True)
        self._row_selected(w.list_stack.currentRow())
        self.schedule()

    def _poll_futures(self):
        if not self._ready and self._load_future.done():
            self._on_ready()
        f = self._scan_future
        if f is not None and f.done():
            self._scan_future = None
            token, s = f.token, f.s
            try:
                G = f.result()
            except Exception as exc:      # noqa: BLE001 — shown, not raised
                self.win.statusBar().showMessage(f"Scan failed: {exc}", 8000)
                return
            if token == self._scan_token and self.view is not None:
                self.view.scan = (s, G, best_s(s, G))
                self.view.scanning = False
                self._show_verdict()
                self.win.propagator_canvas.set_view(self.view)

    # ── the stack ────────────────────────────────────────────────────────
    def _current(self) -> M.Element | None:
        i = self.win.list_stack.currentRow()
        return self.stack[i] if 0 <= i < len(self.stack) else None

    def _refresh_list(self, select=None):
        lst = self.win.list_stack
        row = lst.currentRow() if select is None else select
        lst.blockSignals(True)
        lst.clear()
        for i, e in enumerate(self.stack):
            lst.addItem(_item_text(i, e))
            lst.item(i).setToolTip(f"{e.shelf} / {e.book} / {e.page}")
        lst.setCurrentRow(min(row, len(self.stack) - 1))
        lst.blockSignals(False)
        self._row_selected(lst.currentRow())
        self._update_buttons()

    def _row_selected(self, row):
        w = self.win
        e = self.stack[row] if 0 <= row < len(self.stack) else None
        for wd in (w.cmb_material, w.cmb_page, w.sp_thickness):
            wd.blockSignals(True)
        w.sp_thickness.setEnabled(e is not None)
        if e is not None:
            w.sp_thickness.setValue(e.thickness_mm)
            if self._ready:
                i = w.cmb_material.findData(f"{e.shelf}/{e.book}")
                w.cmb_material.setCurrentIndex(i)
                w.cmb_material.lineEdit().setCursorPosition(0)   # show the start
                self._fill_pages(e)
        for wd in (w.cmb_material, w.cmb_page, w.sp_thickness):
            wd.blockSignals(False)
        self._element_info()

    def _fill_pages(self, e):
        cmb = self.win.cmb_page
        cmb.blockSignals(True)
        cmb.clear()
        for p in M.pages(e.shelf, e.book):
            cmb.addItem(p.page_name or p.page, p.page)
            cmb.setItemData(cmb.count() - 1, p.page_name, Qt.ToolTipRole)
        cmb.setCurrentIndex(max(cmb.findData(e.page), 0))
        cmb.blockSignals(False)

    def _material_typed(self, text):
        i = self.win.cmb_material.findText(text)
        if i >= 0:
            self.win.cmb_material.setCurrentIndex(i)
            self._material_chosen(i)

    def _material_chosen(self, index):
        e = self._current()
        data = self.win.cmb_material.itemData(index)
        if e is None or not data:
            return
        e.shelf, e.book = data.split("/", 1)
        pages = M.pages(e.shelf, e.book)
        # prefer a page with data at the pulse's wavelength
        lam = self._lambda0()
        e.page = pages[0].page if pages else ""
        for p in pages:
            try:
                lo, hi = M.wl_range(M.Element(p.shelf, p.book, p.page))
            except ValueError:
                continue
            if lam is None or lo <= lam <= hi:
                e.page = p.page
                break
        self._fill_pages(e)
        self.win.cmb_material.lineEdit().setCursorPosition(0)
        self._element_changed()

    def _page_chosen(self, index):
        e = self._current()
        page = self.win.cmb_page.itemData(index)
        if e is not None and page:
            e.page = page
            self._element_changed()

    def _thickness_changed(self, value):
        e = self._current()
        if e is not None:
            e.thickness_mm = float(value)
            self._element_changed()

    def _element_changed(self):
        lst = self.win.list_stack
        i = lst.currentRow()
        if 0 <= i < len(self.stack):
            e = self.stack[i]
            lst.item(i).setText(_item_text(i, e))
            lst.item(i).setToolTip(f"{e.shelf} / {e.book} / {e.page}")
        self._element_info()
        self.schedule()

    def add_element(self):
        cur = self._current()
        e = M.Element(cur.shelf, cur.book, cur.page, 1.0) if cur else M.Element()
        self.stack.append(e)
        self._refresh_list(select=len(self.stack) - 1)
        self.schedule()

    def remove_element(self):
        i = self.win.list_stack.currentRow()
        if 0 <= i < len(self.stack):
            del self.stack[i]
            self._refresh_list(select=min(i, len(self.stack) - 1))
            self.schedule()

    def _lambda0(self):
        return None if self.pulse is None else 2 * np.pi * 299.792458 / self.pulse.w0_pulse_rad_fs

    def _element_info(self):
        lbl = self.win.lbl_element_info
        e = self._current()
        if e is None or not self._ready:
            lbl.setText("" if self._ready else "loading materials…")
            return
        lam = self._lambda0() or 800.0
        try:
            lo, hi = M.wl_range(e)
            n0 = float(M.index_function(e)(np.array([lam]))[0])
            w0 = 2 * np.pi * 299.792458 / lam
            gdd = stack_phase([(M.index_function(e), 1.0)], np.array([w0]), w0).gdd
        except ValueError as exc:
            lbl.setText(str(exc))
            return
        lbl.setText(f"n {n0:.4f} · {gdd:.1f} fs²/mm at {lam:.0f} nm\n"
                    f"data {lo:.0f}–{hi:.0f} nm")

    # ── inputs ───────────────────────────────────────────────────────────
    def set_pulse(self, pulse: PulseSource | None):
        self.pulse = pulse
        w = self.win
        if pulse is None:
            w.lbl_prop_pulse.setText("No pulse yet")
            w.lbl_prop_pulse_info.setText("")
        else:
            v = pulse_views(pulse.E, pulse.t_fs, pulse.w_rad_fs, pulse.w0_signal_rad_fs)
            gdd, _ = pulse_dispersion(pulse.E, pulse.w_rad_fs)
            w.lbl_prop_pulse.setText(pulse.source or "retrieved pulse")
            w.lbl_prop_pulse_info.setText(
                f"λ₀ {v['lambda0_nm']:.0f} nm · FWHM {v['fwhm_t']:.1f} fs\n"
                f"GDD {gdd:+.0f} fs² (as retrieved)")
        self._rematch_measured()
        self._element_info()
        self.schedule()

    def load_result_dialog(self):
        start = self.win.settings_ctl.last_dir()
        path, _ = QFileDialog.getOpenFileName(self.win, "Load retrieval result", start,
                                              "Retrieval result (*.npz)")
        if not path:
            return
        try:
            self.set_pulse(load_result(path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self.win, "Could not load", str(exc))

    def load_measured_dialog(self):
        start = self.win.settings_ctl.last_dir()
        path, _ = QFileDialog.getOpenFileName(
            self.win, "Load trace measured after the stack", start,
            "FROG trace (*.dwc *.npz);;Raw trace (*.dwc);;Prepared trace (*.npz)")
        if path:
            self.load_measured(path)

    def load_measured(self, path):
        try:
            if Path(path).suffix.lower() == ".npz":
                self._meas_src = ("npz", load_prepared(path))
            else:
                self._meas_src = ("dwc", read_dwc(path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self.win, "Could not load", str(exc))
            return
        self.win.lbl_prop_meas.setText(Path(path).name)
        self._rematch_measured()
        self.schedule()

    def set_measured_source(self, kind, obj, name=""):
        """Programmatic load (tests): ("dwc", RawTrace) or ("npz", PreparedTrace)."""
        self._meas_src = (kind, obj)
        self.win.lbl_prop_meas.setText(name or kind)
        self._rematch_measured()
        self.schedule()

    def _rematch_measured(self):
        """Put the measured trace onto the current pulse's grid — again
        whenever the pulse changes."""
        self.meas = None
        if self._meas_src is None or self.pulse is None:
            return
        kind, obj = self._meas_src
        try:
            if kind == "npz":
                self.meas = regrid(obj, *self.pulse.grid)
            else:
                self.meas, _ = prepare(obj, self.win.prep.settings(), grid=self.pulse.grid)
        except ValueError as exc:
            self.win.statusBar().showMessage(f"Measured trace: {exc}", 8000)

    # ── compute ──────────────────────────────────────────────────────────
    def schedule(self, *_):
        self._debounce.start()

    def recompute(self):
        w = self.win
        self._scan_token += 1            # whatever is running is now stale
        if self.pulse is None or not self._ready:
            self.view = None
            w.propagator_canvas.set_view(None, "Run a retrieval or load a saved result"
                                         if self._ready else "Loading materials…")
            w.lbl_stack_info.setText("")
            self._show_verdict()
            return
        p = self.pulse
        w_abs = p.w0_pulse_rad_fs + p.w_rad_fs
        try:
            sp = stack_phase([(M.index_function(e), e.thickness_mm) for e in self.stack],
                             w_abs, p.w0_pulse_rad_fs)
        except ValueError as exc:
            w.lbl_stack_info.setText(str(exc))
            return
        mode = w.cmb_disp_mode.currentData() or "full"
        phase = sp.applied(mode)
        self._phase = phase
        E_as = propagate(p.E, phase)
        E_rev = propagate(time_reverse(p.E), phase)
        view = PropView(pulse=p, E_as=E_as, E_rev=E_rev, T_as=shg_trace(E_as),
                        T_rev=shg_trace(E_rev), gdd=sp.gdd, meas=self.meas)
        # spectral energy that fell outside a page's data
        I = np.abs(t2w(p.E)) ** 2
        I /= I.sum()
        outside = [e.book for e, part in zip(self.stack, sp.elements)
                   if I[part.filled].sum() > COVERAGE_WARN]
        gdd_p, _ = pulse_dispersion(p.E, p.w_rad_fs)
        info = (f"stack GDD {sp.gdd:+.1f} fs² · TOD {sp.tod:+.0f} fs³\n"
                f"pulse GDD {gdd_p:+.0f} → {gdd_p + sp.gdd:+.0f} fs²")
        if outside:
            info += "\n⚠ outside data range: " + ", ".join(outside)
        w.lbl_stack_info.setText(info)
        if self.meas is not None:
            mask = self.meas.mask
            view.G_as = trace_errors(self.meas.trace, view.T_as, mask)[0]
            view.G_rev = trace_errors(self.meas.trace, view.T_rev, mask)[0]
            rng = w.sp_scan_range.value()
            s = np.linspace(-rng, rng, int(w.sp_scan_points.value()))
            view.scanning = True
            f = self._pool.submit(scan, p.E, phase, self.meas.trace, s, mask)
            f.token, f.s = self._scan_token, s
            self._scan_future = f
        self.view = view
        self._show_verdict()
        w.propagator_canvas.set_view(view)

    def wait_scan(self, timeout=60):
        """Block until the running scan is in (tests)."""
        f = self._scan_future
        if f is not None:
            cf.wait([f], timeout)
        self._poll_futures()

    def _show_verdict(self):
        lbl = self.win.lbl_verdict
        v = self.view
        if v is None or v.G_as is None:
            lbl.setText("No measured trace yet")
            return
        lines = [f"→ {verdict(v.G_as, v.G_rev)}",
                 f"G {v.G_as:.2e} · rev. {v.G_rev:.2e}"]
        if v.scan is not None:
            b = v.scan[2]
            lines.append(f"best fit {b:+.2f}× ({b * v.gdd:+.0f} fs²)")
        elif v.scanning:
            lines.append("scanning…")
        lbl.setText("\n".join(lines))

    def _update_buttons(self):
        self.win.btn_stack_remove.setEnabled(len(self.stack) > 0)
