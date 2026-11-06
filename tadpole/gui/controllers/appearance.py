"""Theme switching, trace colormap/scale, and the save-picture button."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFileDialog

from tadpole.design.icons import MOON_ICON, PICTURE_ICON, SUN_ICON
from tadpole.design.theme import install_theme


class AppearanceController:
    def __init__(self, win, theme: str):
        self.win = win
        self.theme = theme
        win.btn_theme.clicked.connect(self.toggle_theme)
        win.btn_picture.clicked.connect(self.save_picture)
        win.btn_graphics.clicked.connect(self.show_graphics)
        win.dlg_graphics.changed.connect(self.apply_graphics)
        self._refresh_icons()

    def canvases(self):
        return (self.win.prep_canvas, self.win.recon_canvas)

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        install_theme(QApplication.instance(), self.theme)
        for c in self.canvases():
            c.restyle()
        for lamp in (self.win.engine_lamp,):
            lamp.update()
        self._refresh_icons()

    def _refresh_icons(self):
        # The icon shows the theme a click switches TO (Lillypad convention).
        icon = SUN_ICON if self.theme == "dark" else MOON_ICON
        self.win.btn_theme.setIcon(QIcon(str(icon)))
        self.win.btn_picture.setIcon(QIcon(str(PICTURE_ICON[self.theme])))

    def show_graphics(self):
        c = self.win.prep_canvas
        self.win.dlg_graphics.set_values(c.cmap_name, c.cmap_reversed, c.log_scale)
        self.win.dlg_graphics.show()
        self.win.dlg_graphics.raise_()

    def apply_graphics(self, cmap, reversed_, log):
        for c in self.canvases():
            c.set_appearance(cmap, reversed_, log)

    def save_picture(self):
        canvas = self.win.current_canvas()
        page = "prepare" if canvas is self.win.prep_canvas else "reconstruct"
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        start = str(Path(self.win.settings_ctl.last_dir()) / f"tadpole_{page}_{stamp}.png")
        path, _ = QFileDialog.getSaveFileName(self.win, "Save picture", start,
                                              "PNG image (*.png)")
        if path:
            canvas.save_picture(path)
            self.win.statusBar().showMessage(f"Saved {path}", 5000)
