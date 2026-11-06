"""Graphics settings: colormap, its direction, and linear/log trace scale.

A non-modal Qt.Tool window, like Lillypad's dialogs; changes apply live.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QPushButton, QVBoxLayout)

from tadpole.design.mpl_style import TRACE_COLORMAPS
from tadpole.gui.widgets import ComboBox


class GraphicsDialog(QDialog):
    changed = Signal(str, bool, bool)           # cmap, reversed, log

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Graphics Settings")
        lay = QVBoxLayout(self)
        box = QGroupBox("Trace display")
        form = QFormLayout(box)
        self.cmap = ComboBox()
        self.cmap.addItems(TRACE_COLORMAPS)
        self.reversed = QCheckBox("Reverse colormap")
        self.log = QCheckBox("Logarithmic scale (3 decades)")
        form.addRow("Colormap", self.cmap)
        form.addRow(self.reversed)
        form.addRow(self.log)
        lay.addWidget(box)
        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        row.addWidget(close)
        lay.addLayout(row)
        self.cmap.currentTextChanged.connect(self._emit)
        self.reversed.toggled.connect(self._emit)
        self.log.toggled.connect(self._emit)

    def _emit(self, *_):
        self.changed.emit(self.cmap.currentText(), self.reversed.isChecked(),
                          self.log.isChecked())

    def set_values(self, cmap, reversed_, log):
        for w in (self.cmap, self.reversed, self.log):
            w.blockSignals(True)
        if cmap in TRACE_COLORMAPS:
            self.cmap.setCurrentText(cmap)
        self.reversed.setChecked(bool(reversed_))
        self.log.setChecked(bool(log))
        for w in (self.cmap, self.reversed, self.log):
            w.blockSignals(False)
