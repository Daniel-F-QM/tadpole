"""Small reusable widgets, taken from Lillypad (lillypad/gui/widgets.py).

Value widgets refuse the mouse wheel (scrolling the side panel must never edit
a setting) and number boxes always write "." as the decimal separator. Colour
comes from design.tokens.PALETTE, read at paint time, so a theme switch
reaches widgets that already exist.
"""

from __future__ import annotations

from PySide6.QtCore import QLocale, QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QLabel,
                               QSlider, QSpinBox, QWidget)

from tadpole.design.tokens import PALETTE


class _NoWheel:
    def wheelEvent(self, ev):
        ev.ignore()


def number_locale() -> QLocale:
    """"." as the decimal separator and no group separator, whatever the
    Windows region says."""
    loc = QLocale.c()
    loc.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator
                         | QLocale.NumberOption.RejectGroupSeparator)
    return loc


def use_dot_decimals() -> None:
    QLocale.setDefault(number_locale())


class _DotDecimal:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(number_locale())


class SpinBox(_NoWheel, _DotDecimal, QSpinBox):             pass
class DoubleSpinBox(_NoWheel, _DotDecimal, QDoubleSpinBox): pass
class ComboBox(_NoWheel, QComboBox):                       pass
class Slider(_NoWheel, QSlider):                           pass


class OptionalDoubleSpinBox(DoubleSpinBox):
    """A number box whose MINIMUM means "not set" and shows `auto_text`.

    For crop bounds and similar optional values: value() at the minimum reads
    back as None through `optional_value()`.
    """
    def __init__(self, lo, hi, auto_text="auto", decimals=1, step=1.0, parent=None):
        super().__init__(parent)
        self.setDecimals(decimals)
        self.setRange(lo, hi)
        self.setSingleStep(step)
        self.setSpecialValueText(auto_text)
        self.setValue(lo)

    def optional_value(self):
        v = self.value()
        return None if v <= self.minimum() else float(v)

    def set_optional_value(self, v):
        self.setValue(self.minimum() if v is None else float(v))


def _hline():
    f = QFrame(); f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    return f


def dim_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("dim")
    lbl.setWordWrap(True)
    return lbl


class StatusLamp(QWidget):
    """Round status indicator with a soft halo (Lillypad's lamp).

    States: unknown (grey), ok (green), warn (amber), sat (red), busy (accent).
    """
    COLORS = {"unknown": "text_disabled", "ok": "good", "warn": "warn",
              "sat": "danger", "busy": "accent"}

    def __init__(self, diameter=12, parent=None):
        super().__init__(parent)
        self._d = diameter
        self._state = "unknown"
        self.setFixedSize(diameter + 8, diameter + 8)

    def set_state(self, state):
        if state not in self.COLORS:
            raise ValueError(f"unknown lamp state {state!r}")
        if state != self._state:
            self._state = state
            self.update()

    def state(self):
        return self._state

    def paintEvent(self, _event):
        color = QColor(PALETTE[self.COLORS[self._state]])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = self.width() / 2.0, self.height() / 2.0, self._d / 2.0
        if self._state != "unknown":
            halo = QColor(color)
            halo.setAlpha(70)
            p.setPen(Qt.NoPen); p.setBrush(halo)
            p.drawEllipse(QPointF(cx, cy), r + 3.0, r + 3.0)
        p.setPen(QPen(color.darker(150), 1.0))
        p.setBrush(color)
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.end()
