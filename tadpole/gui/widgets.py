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
                               QLineEdit, QSlider, QSpinBox, QWidget)

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
class ComboBox(_NoWheel, QComboBox):
    """Sized by the panel, not by its longest item: the fixed-width side
    column clips anything wider. The open list still shows full names."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.view().setTextElideMode(Qt.ElideNone)

    def showPopup(self):
        # the list may be wider than the box, so long names stay readable
        self.view().setMinimumWidth(self.view().sizeHintForColumn(0) + 24)
        super().showPopup()


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


class AxisBoundEditor(QLineEdit):
    """The box that opens over an axis' end tick label when it is clicked
    (Lillypad's). A child of the canvas, so it sits on top of the plot.

    Return commits, Escape or clicking away cancels: a half-typed number that
    is abandoned should never land on an axis.
    """
    PAD = 10
    MIN_W = 56

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._target = None       # (ax, "x"|"y", "lo"|"hi") while open
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self.returnPressed.connect(self._commit)
        self.editingFinished.connect(self.cancel)

    def open_on(self, ax, which, end, value, rect):
        self._target = (ax, which, end)
        # %g, not the tick's rounded label, so "nudge the max" moves nothing else
        self.setText(f"{value:g}")
        w = max(rect.width() + self.PAD, self.MIN_W)
        self.setGeometry(rect.center().x() - w // 2, rect.top() - 2,
                         w, rect.height() + 6)
        self.show()
        self.raise_()
        self.setFocus(Qt.OtherFocusReason)
        self.selectAll()

    def cancel(self):
        self._target = None
        self.hide()

    def _commit(self):
        if self._target is None:
            return
        ax, which, end = self._target
        text = self.text().strip()
        # cleared first: applying redraws, which can re-enter through cancel()
        self._target = None
        self.hide()
        try:
            value = float(text)
        except ValueError:
            hint = " Use '.' for decimals." if "," in text else ""
            self._canvas.message.emit(f"{text!r} is not a number — axis unchanged.{hint}")
            return
        self._canvas.apply_axis_bound(ax, which, end, value)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancel()
            return
        super().keyPressEvent(event)


class ElidedLabel(QLabel):
    """One line that shortens itself with "…" to the width it is given; the
    full text is in the tooltip. For file names and the like in the fixed-
    width side panel, which must neither wrap onto more lines nor widen it."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setMinimumWidth(20)
        self.setText(text)

    def setText(self, text):
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()

    def text(self):
        return self._full

    def _elide(self):
        w = max(self.width(), 20)
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, w))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def minimumSizeHint(self):
        h = super().minimumSizeHint()
        h.setWidth(20)
        return h

    def sizeHint(self):
        h = super().sizeHint()
        h.setWidth(min(h.width(), 200))
        return h
