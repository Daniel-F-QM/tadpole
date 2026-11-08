"""TadpoleWindow: wiring only. Widgets come from build_ui, behaviour from the
controllers (Lillypad's split).

Construction order: state → dialogs → widgets → controllers → restore
settings → start the Julia engine (in the background; the window is usable
while Julia boots).
"""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow

from tadpole.gui.controllers.appearance import AppearanceController
from tadpole.gui.controllers.persistence import SettingsController
from tadpole.gui.controllers.prep import PrepController
from tadpole.gui.controllers.propagator import PropagatorController
from tadpole.gui.controllers.recon import ReconController
from tadpole.gui.dialogs.graphics import GraphicsDialog
from tadpole.gui.state import AppState
from tadpole.gui.window.build_ui import build_ui
from tadpole.settings import load_settings


class TadpoleWindow(QMainWindow):
    def __init__(self, theme: str = "dark", engine=None, start_engine: bool = True):
        super().__init__()
        self.setWindowTitle("Tadpole — FROG reconstruction")
        # fits a 1080p screen at 150 % scaling (1280×720 logical)
        self.setMinimumSize(1000, 600)
        self.state = AppState()
        self.state_last_dir = ""
        self.dlg_graphics = GraphicsDialog(self)
        build_ui(self)
        self.settings_ctl = SettingsController(self)
        self.appearance = AppearanceController(self, theme)
        self.prep = PrepController(self)
        self.recon = ReconController(self, engine)
        self.propagator = PropagatorController(self)
        self.mode_group.idClicked.connect(self.set_mode)
        for c in self.canvases():
            c.message.connect(lambda text: self.statusBar().showMessage(text, 6000))
        self.settings_ctl.restore()
        if start_engine:
            self.recon.start_engine()

    # ── pages ────────────────────────────────────────────────────────────
    def set_mode(self, index: int):
        self.stack.setCurrentIndex(index)
        self.mode_group.button(index).setChecked(True)

    PAGE_NAMES = ("prepare", "reconstruct", "analysis", "propagator")

    def canvases(self):
        return (self.prep_canvas, self.recon_canvas, self.analysis_canvas,
                self.propagator_canvas)

    def current_canvas(self):
        return self.canvases()[self.stack.currentIndex()]

    def current_page_name(self) -> str:
        return self.PAGE_NAMES[self.stack.currentIndex()]

    # ── show / close ─────────────────────────────────────────────────────
    def show_restored(self):
        g = load_settings().get("window", {})
        try:
            self.resize(int(g.get("w", 1280)), int(g.get("h", 800)))
        except (TypeError, ValueError):
            pass
        self.showMaximized()

    def closeEvent(self, event):
        self.settings_ctl.save()
        self.recon.shutdown()
        self.propagator.shutdown()
        super().closeEvent(event)
