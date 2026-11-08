"""Advanced settings: the rarely touched options of a page, kept out of the
side panel so it never has to scroll.

A non-modal Qt.Tool window like Graphics Settings. The widgets are created by
build_ui and attached to the window under their usual names, so controllers
and settings persistence don't care where they are shown. Changes apply live.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)


class AdvancedDialog(QDialog):
    def __init__(self, title, rows, parent=None):
        """`rows`: (label, widget, tooltip). Returns the row labels in
        `self.labels` (keyed by widget) so callers can hide a row."""
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        box = QGroupBox("Advanced")
        form = QFormLayout(box)
        form.setContentsMargins(8, 14, 8, 8)
        form.setVerticalSpacing(6)
        self.labels = {}
        for label, widget, tip in rows:
            lbl = QLabel(label)
            if tip:
                lbl.setToolTip(tip)
                widget.setToolTip(tip)
            form.addRow(lbl, widget)
            self.labels[widget] = lbl
        lay.addWidget(box)
        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        row.addWidget(close)
        lay.addLayout(row)

    def show_near(self, anchor):
        """Open beside `anchor` (the button that asked), on top."""
        self.adjustSize()
        pos = anchor.mapToGlobal(anchor.rect().topRight())
        self.move(pos.x() + 8, max(0, pos.y() - self.height() + anchor.height()))
        self.show()
        self.raise_()
        self.activateWindow()
