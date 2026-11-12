"""Building the window's widgets — free functions, like Lillypad's build_ui.

Layout: a toolbar, then per page a fixed-width control column of QGroupBoxes
on the left and the figure on the right. Three pages — Prepare, Reconstruct,
Analysis — in a QStackedWidget switched by the toolbar's mode buttons.

The column never scrolls: explanations are tooltips, paired values share a
row, and rarely touched options live in each page's Advanced… window
(dialogs/advanced.py). Widgets are attached to `win` by name wherever they
are shown; the controllers wire them.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QCheckBox,
                               QComboBox, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QStatusBar, QToolBar, QVBoxLayout,
                               QWidget)

from tadpole.engine.protocol import ALGORITHMS
from tadpole.gui.canvas.analysis_canvas import AnalysisCanvas
from tadpole.gui.canvas.prep_canvas import PrepCanvas
from tadpole.gui.canvas.propagator_canvas import PropagatorCanvas
from tadpole.gui.canvas.recon_canvas import ReconCanvas
from tadpole.gui.dialogs.advanced import AdvancedDialog
from tadpole.gui.widgets import (ComboBox, DoubleSpinBox, ElidedLabel,
                                 OptionalDoubleSpinBox, SpinBox, StatusLamp,
                                 dim_label)

PANEL_W = 280

ALGORITHM_NOTES = {
    "pcgpa": "Principal-component generalized projections. Fast and robust; "
             "needs the square N×N grid.",
    "gpa": "Generalized projections with an exact line search on the "
           "mathematical-form constraint. Slower per iteration than PCGPA.",
    "ptychographic": "One measured spectrum per update; works on the measured "
                     "delays and with missing data. Not implemented yet.",
}

ERROR_HELP = ("G: rms trace error (Trebino). G′: the same residual normalised by "
              "the measured trace's norm, so it does not depend on grid size.")


class _Column(QScrollArea):
    """The fixed-width control column.

    Laid out to fit a maximized window on a 1280×720 logical screen (1080p at
    150 %) with a taskbar: a ~550 px tall column, in the real windows11 style
    (the tests measure exactly that). The scroll area is only a safety net
    for an even shorter window: without it Qt squeezes every row below its
    height and clips the text instead. It never scrolls sideways: the
    contents always get exactly the column's width.

    `#panel` scopes the stylesheet's compact rules (slimmer buttons and
    group boxes) to the side columns.
    """
    MIN_FIELD_W = 40     # inputs never ask for more than this by themselves

    def __init__(self, inner: QWidget):
        super().__init__()
        self.inner = inner
        inner.setObjectName("panel")
        # A spin box sizes itself for its longest possible value (−100000.0
        # fs…) and a combo box for its longest item; in a fixed-width column
        # the layout decides instead.
        for wd in inner.findChildren(QWidget):
            if isinstance(wd, (QAbstractSpinBox, QComboBox)):
                wd.setMinimumWidth(self.MIN_FIELD_W)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFixedWidth(PANEL_W)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.inner.setFixedWidth(self.viewport().width())

    def needs_scrolling(self) -> bool:
        return self.inner.minimumSizeHint().height() > self.viewport().height()

    def too_wide(self) -> bool:
        return self.inner.layout().minimumSize().width() > self.viewport().width()


def _column(*groups, bottom=None) -> _Column:
    col = QWidget()
    lay = QVBoxLayout(col)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for g in groups:
        lay.addWidget(g)
    lay.addStretch()
    if bottom is not None:
        lay.addWidget(bottom)
    return _Column(col)


def _form(box: QGroupBox) -> QFormLayout:
    f = QFormLayout(box)
    f.setContentsMargins(6, 12, 6, 6)
    f.setHorizontalSpacing(8)
    f.setVerticalSpacing(3)
    f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return f


def _vbox(box: QGroupBox) -> QVBoxLayout:
    v = QVBoxLayout(box)
    v.setContentsMargins(6, 12, 6, 6)
    v.setSpacing(3)
    return v


def _pair(a, b, sep="–") -> QWidget:
    """Two value widgets on one row, e.g. a min–max range."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    for x in (a, b):
        x.setMinimumWidth(40)      # not the width of their longest possible text
    h.addWidget(a, 1)
    lbl = QLabel(sep)
    lbl.setObjectName("dim")
    h.addWidget(lbl)
    h.addWidget(b, 1)
    return w


def _row(*widgets) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(4)
    for w in widgets:
        h.addWidget(w, 1)
    return h


def _adv_button(name) -> QPushButton:
    b = QPushButton("Advanced…")
    b.setObjectName("compact")
    b.setToolTip(f"Rarely changed {name} settings")
    return b


def build_toolbar(win) -> None:
    tb = QToolBar()
    tb.setMovable(False)
    tb.setIconSize(QSize(18, 18))
    win.addToolBar(tb)
    win.btn_mode_prep = QPushButton("Prepare")
    win.btn_mode_recon = QPushButton("Reconstruct")
    win.btn_mode_analysis = QPushButton("Analysis")
    win.btn_mode_analysis.setToolTip("Convergence and residual of the last "
                                     "retrieval — available once one has finished")
    win.btn_mode_analysis.setEnabled(False)
    win.btn_mode_prop = QPushButton("Propagator")
    win.btn_mode_prop.setToolTip("Put the retrieved pulse through material and "
                                 "compare with a trace measured after it")
    win.mode_group = QButtonGroup(win)
    for i, b in enumerate((win.btn_mode_prep, win.btn_mode_recon,
                           win.btn_mode_analysis, win.btn_mode_prop)):
        b.setObjectName("mode")
        b.setCheckable(True)
        win.mode_group.addButton(b, i)
        tb.addWidget(b)
    win.btn_mode_prep.setChecked(True)
    tb.addSeparator()
    win.btn_open = QPushButton("Open .dwc…")
    win.btn_simulate = QPushButton("Simulate")
    win.btn_simulate.setToolTip("Load a synthetic chirped-pulse trace with "
                                "background and noise")
    win.btn_graphics = QPushButton("Graphics Settings")
    for b in (win.btn_open, win.btn_simulate, win.btn_graphics):
        tb.addWidget(b)
    spacer = QWidget()
    spacer.setObjectName("spacer")
    spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    tb.addWidget(spacer)
    win.btn_theme = QPushButton()
    win.btn_theme.setFixedWidth(42)
    win.btn_theme.setIconSize(QSize(18, 18))
    win.btn_theme.setToolTip("Switch light / dark theme")
    win.btn_picture = QPushButton()
    win.btn_picture.setFixedWidth(42)
    win.btn_picture.setIconSize(QSize(18, 18))
    win.btn_picture.setToolTip("Save the current page's plots as a PNG")
    tb.addWidget(win.btn_theme)
    tb.addWidget(win.btn_picture)


def build_prep_page(win) -> QWidget:
    # File
    g_file = QGroupBox("File")
    v = _vbox(g_file)
    win.lbl_file = ElidedLabel("No trace loaded")
    win.lbl_file_info = ElidedLabel("")
    win.lbl_file_info.setObjectName("dim")
    win.lbl_file_info.setVisible(False)        # until a trace is loaded
    win.btn_file_open = QPushButton("Open trace (.dwc)…")
    win.btn_file_open.setObjectName("accentcompact")
    win.btn_file_open.setToolTip("Open a FROG trace recorded by Lillypad")
    v.addWidget(win.lbl_file)
    v.addWidget(win.lbl_file_info)
    v.addWidget(win.btn_file_open)

    # Background
    g_bg = QGroupBox("Background")
    f = _form(g_bg)
    win.cmb_bg = ComboBox()
    win.cmb_bg.addItem("None", "none")
    win.cmb_bg.addItem("Constant (edge mean)", "constant")
    win.cmb_bg.addItem("Spectral (edge spectrum)", "spectral")
    win.cmb_bg.setToolTip("Negative pixels are kept until the trace is binned onto "
                          "the grid, so noise averages out.")
    win.sp_edge_cols = SpinBox(); win.sp_edge_cols.setRange(0, 200)
    win.sp_edge_cols.setToolTip("Columns at each end of the delay axis used as "
                                "background.")
    f.addRow("Mode", win.cmb_bg)
    f.addRow("Edge delays", win.sp_edge_cols)

    # Crop and delay zero
    g_crop = QGroupBox("Crop and delay zero")
    f = _form(g_crop)
    win.sp_wl_min = OptionalDoubleSpinBox(0.0, 10000.0, "full", 1, 1.0)
    win.sp_wl_max = OptionalDoubleSpinBox(0.0, 10000.0, "full", 1, 1.0)
    win.sp_d_min = OptionalDoubleSpinBox(-100000.0, 100000.0, "full", 1, 5.0)
    win.sp_d_max = OptionalDoubleSpinBox(-100000.0, 100000.0, "full", 1, 5.0)
    for sp, tip in ((win.sp_wl_min, "Shortest wavelength kept (nm)"),
                    (win.sp_wl_max, "Longest wavelength kept (nm)"),
                    (win.sp_d_min, "Earliest delay kept (fs)"),
                    (win.sp_d_max, "Latest delay kept (fs)")):
        sp.setToolTip(tip + ". At the minimum: no crop.")
    f.addRow("λ (nm)", _pair(win.sp_wl_min, win.sp_wl_max))
    f.addRow("τ (fs)", _pair(win.sp_d_min, win.sp_d_max))
    win.cmb_zero = ComboBox()
    win.cmb_zero.addItem("Centroid of marginal", "centroid")
    win.cmb_zero.addItem("Peak of marginal", "peak")
    win.cmb_zero.addItem("As in file (centre column)", "none")
    win.chk_sym = QCheckBox("Symmetrize in delay (SHG)")
    win.cmb_zero.setToolTip("How delay zero is found. Extra shift: Advanced…")
    f.addRow("Zero from", win.cmb_zero)
    f.addRow(win.chk_sym)

    # Fourier grid
    g_grid = QGroupBox("Fourier grid")
    f = _form(g_grid)
    win.cmb_n = ComboBox()
    for n in (64, 128, 256, 512, 1024):
        win.cmb_n.addItem(str(n), n)
    win.cmb_n.setToolTip("Grid size N×N. dω = 2π/(N·dt).")
    win.lbl_grid = QLabel("")
    win.lbl_grid.setObjectName("readout_sm")
    win.lbl_grid.setWordWrap(True)
    win.lbl_warn = QLabel("")
    win.lbl_warn.setObjectName("satwarn")
    win.lbl_warn.setWordWrap(True)
    win.lbl_warn.setVisible(False)
    f.addRow("N", win.cmb_n)
    f.addRow(win.lbl_grid)
    f.addRow(win.lbl_warn)

    # Output
    g_out = QGroupBox("Output")
    v = _vbox(g_out)
    win.btn_send = QPushButton("Send to Reconstructor")
    win.btn_send.setObjectName("accent")
    win.btn_save_prep = QPushButton("Save .npz…")
    win.btn_save_prep.setObjectName("compact")
    win.btn_save_prep.setToolTip("Save the prepared trace (.npz)")
    v.addWidget(win.btn_send)

    # Advanced (pop-up)
    win.sp_edge_rows = SpinBox(); win.sp_edge_rows.setRange(0, 500)
    win.sp_zero_off = DoubleSpinBox(); win.sp_zero_off.setRange(-10000, 10000)
    win.sp_zero_off.setDecimals(2); win.sp_zero_off.setSuffix(" fs")
    win.sp_dt = DoubleSpinBox(); win.sp_dt.setRange(0.0, 1000.0)
    win.sp_dt.setDecimals(3); win.sp_dt.setSingleStep(0.1)
    win.sp_dt.setSpecialValueText("auto"); win.sp_dt.setSuffix(" fs")
    win.dlg_prep_adv = AdvancedDialog("Prepare — advanced", [
        ("Edge λ rows", win.sp_edge_rows,
         "Rows at each end of the wavelength axis used to remove an additional "
         "per-delay offset. 0 = off."),
        ("Extra delay shift", win.sp_zero_off,
         "Added to the automatically found delay zero."),
        ("Grid dt", win.sp_dt,
         "Time step of the Fourier grid. Auto leaves the same headroom on the "
         "delay and the frequency window."),
    ], win)
    win.btn_prep_adv = _adv_button("preparation")
    v.addLayout(_row(win.btn_save_prep, win.btn_prep_adv))

    win.prep_canvas = PrepCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    win.prep_column = _column(g_file, g_bg, g_crop, g_grid, g_out)
    h.addWidget(win.prep_column)
    h.addWidget(win.prep_canvas, 1)
    return page


def build_recon_page(win) -> QWidget:
    g_in = QGroupBox("Input")
    v = _vbox(g_in)
    win.lbl_recon_input = ElidedLabel("Nothing prepared yet")
    win.lbl_recon_input_info = dim_label("")
    win.btn_load_prep = QPushButton("Load prepared (.npz)…")
    win.btn_load_prep.setObjectName("compact")
    v.addWidget(win.lbl_recon_input)
    v.addWidget(win.lbl_recon_input_info)
    v.addWidget(win.btn_load_prep)

    g_alg = QGroupBox("Algorithm")
    f = _form(g_alg)
    win.cmb_alg = ComboBox()
    for key, name in ALGORITHMS.items():
        win.cmb_alg.addItem(name, key)
    win.cmb_init = ComboBox()
    win.cmb_init.addItem("Autocorrelation", "autocorrelation")
    win.cmb_init.addItem("Gaussian", "gaussian")
    win.cmb_init.addItem("Random", "random")
    win.sp_max_iter = SpinBox(); win.sp_max_iter.setRange(1, 1000000)
    win.sp_max_iter.setSingleStep(100)
    win.sp_tol_exp = SpinBox(); win.sp_tol_exp.setRange(-12, -1)
    win.sp_tol_exp.setPrefix("G < 10^")
    win.sp_tol_exp.setToolTip("Stop once G falls below this. " + ERROR_HELP)
    f.addRow(win.cmb_alg)
    f.addRow("Initial guess", win.cmb_init)
    f.addRow("Max iterations", win.sp_max_iter)
    f.addRow("Tolerance", win.sp_tol_exp)

    g_run = QGroupBox("Run")
    v = _vbox(g_run)
    row = QHBoxLayout()
    win.btn_start = QPushButton("Start")
    win.btn_start.setObjectName("accent")
    win.btn_stop = QPushButton("Stop")
    win.btn_stop.setObjectName("danger")
    row.addWidget(win.btn_start)
    row.addWidget(win.btn_stop)
    v.addLayout(row)
    # Text under the bar, not on it: over the accent fill it is unreadable.
    win.progress = QProgressBar()
    win.progress.setFixedHeight(6)
    win.progress.setTextVisible(False)
    win.progress.setRange(0, 100)
    win.progress.setValue(0)
    win.lbl_progress = QLabel("idle")
    win.lbl_progress.setObjectName("dim")
    win.lbl_progress.setToolTip("The current G and G′ are shown on the Retrieved "
                                "trace. " + ERROR_HELP)
    v.addWidget(win.progress)
    v.addWidget(win.lbl_progress)

    g_res = QGroupBox("Result")
    v = _vbox(g_res)
    win.lbl_result = dim_label("No result yet")
    win.btn_save_result = QPushButton("Save result (.npz / .csv)…")
    win.btn_save_result.setObjectName("compact")
    v.addWidget(win.lbl_result)
    v.addWidget(win.btn_save_result)

    # Advanced (pop-up)
    win.sp_seed = SpinBox(); win.sp_seed.setRange(0, 999999)
    win.sp_alpha_min = DoubleSpinBox(); win.sp_alpha_min.setRange(0.0, 2.0)
    win.sp_alpha_max = DoubleSpinBox(); win.sp_alpha_max.setRange(0.0, 2.0)
    for sp in (win.sp_alpha_min, win.sp_alpha_max):
        sp.setDecimals(2); sp.setSingleStep(0.05)
    win.sp_gamma = DoubleSpinBox(); win.sp_gamma.setRange(0.0, 1.0)
    win.sp_gamma.setDecimals(7); win.sp_gamma.setSingleStep(1e-6)
    win.sp_gamma.setSpecialValueText("off")
    win.dlg_recon_adv = AdvancedDialog("Reconstruct — advanced", [
        ("Seed", win.sp_seed, "Random-number seed for the 'Random' initial guess."),
        ("α min", win.sp_alpha_min, "Ptychographic: smallest update step."),
        ("α max", win.sp_alpha_max, "Ptychographic: largest update step."),
        ("γ threshold", win.sp_gamma, "Ptychographic: regularisation threshold."),
    ], win)
    lbl = win.dlg_recon_adv.labels
    win.ptycho_rows = [(lbl[w], w) for w in (win.sp_alpha_min, win.sp_alpha_max,
                                             win.sp_gamma)]
    win.btn_recon_adv = _adv_button("retrieval")

    win.recon_canvas = ReconCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    win.recon_column = _column(g_in, g_alg, g_run, g_res, bottom=win.btn_recon_adv)
    h.addWidget(win.recon_column)
    h.addWidget(win.recon_canvas, 1)
    return page


def build_analysis_page(win) -> QWidget:
    g_sum = QGroupBox("Last retrieval")
    v = _vbox(g_sum)
    win.lbl_analysis = QLabel("No result yet")
    win.lbl_analysis.setObjectName("readout_sm")
    win.lbl_analysis.setWordWrap(True)
    win.lbl_analysis.setToolTip(ERROR_HELP)
    win.btn_save_analysis = QPushButton("Save result (.npz / .csv)…")
    win.btn_save_analysis.setObjectName("compact")
    v.addWidget(win.lbl_analysis)
    v.addWidget(win.btn_save_analysis)

    g_help = QGroupBox("Plots")
    v = _vbox(g_help)
    v.addWidget(dim_label("Drag to zoom, right-click to reset, click an end tick "
                          "label to type a bound. On a colour bar: drag or scroll "
                          "to set the range."))

    win.analysis_canvas = AnalysisCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    win.analysis_column = _column(g_sum, g_help)
    h.addWidget(win.analysis_column)
    h.addWidget(win.analysis_canvas, 1)
    return page


def build_propagator_page(win) -> QWidget:
    # Pulse
    g_pulse = QGroupBox("Pulse")
    v = _vbox(g_pulse)
    win.lbl_prop_pulse = ElidedLabel("No pulse yet")
    win.lbl_prop_pulse_info = dim_label("")
    win.btn_prop_load_result = QPushButton("Load…")
    win.btn_prop_load_result.setObjectName("compact")
    win.btn_prop_load_result.setToolTip("Use a saved retrieval result (.npz) "
                                        "instead of the last one")
    top = QHBoxLayout()
    top.setSpacing(4)
    top.addWidget(win.lbl_prop_pulse, 1)
    top.addWidget(win.btn_prop_load_result)
    v.addLayout(top)
    v.addWidget(win.lbl_prop_pulse_info)

    # Material stack
    g_stack = QGroupBox("Material stack")
    f = _form(g_stack)
    win.list_stack = QListWidget()
    win.list_stack.setFixedHeight(64)
    win.list_stack.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    win.list_stack.setTextElideMode(Qt.ElideMiddle)
    win.list_stack.setToolTip("The elements the pulse passes through, in order. "
                              "Select one to edit it.")
    win.btn_stack_add = QPushButton("Add")
    win.btn_stack_remove = QPushButton("Remove")
    for b in (win.btn_stack_add, win.btn_stack_remove):
        b.setObjectName("compact")
    win.btn_stack_add.setToolTip("Add an element: the material typed in the box, "
                                 "or a copy of the selected one")
    # no focus: a click must not take it from the material box first, which
    # would apply a typed search to the selected element (propagator.py)
    win.btn_stack_add.setFocusPolicy(Qt.NoFocus)
    win.cmb_material = ComboBox()
    win.cmb_material.setEditable(True)
    win.cmb_material.setInsertPolicy(QComboBox.NoInsert)
    win.cmb_material.setToolTip("Type to search refractiveindex.info: material, "
                                "glass or data set, several words narrow it down "
                                "(e.g. SiO2, N-BK7, sapphire, CaF2 li)")
    win.cmb_page = ComboBox()
    win.cmb_page.setToolTip("The data set (author, year, wavelength range)")
    win.sp_thickness = DoubleSpinBox()
    win.sp_thickness.setRange(0.0, 100000.0)
    win.sp_thickness.setDecimals(3)
    win.sp_thickness.setSingleStep(0.5)
    win.sp_thickness.setSuffix(" mm")
    win.lbl_element_info = dim_label("")
    side = QVBoxLayout()
    side.setSpacing(4)
    side.addWidget(win.btn_stack_add)
    side.addWidget(win.btn_stack_remove)
    side.addStretch()
    stack_row = QHBoxLayout()
    stack_row.setSpacing(4)
    stack_row.addWidget(win.list_stack, 1)
    stack_row.addLayout(side)
    f.addRow(stack_row)
    f.addRow("Material", win.cmb_material)
    f.addRow("Data", win.cmb_page)
    f.addRow("Thickness", win.sp_thickness)
    f.addRow(win.lbl_element_info)

    # Dispersion
    g_disp = QGroupBox("Dispersion")
    f = _form(g_disp)
    win.cmb_disp_mode = ComboBox()
    win.cmb_disp_mode.setToolTip("Full n(ω): the exact material phase. GDD only / "
                                 "GDD + TOD: its Taylor terms at the pulse centre. "
                                 "Group delay is always removed.")
    win.lbl_stack_info = QLabel("")
    win.lbl_stack_info.setObjectName("readout_sm")
    win.lbl_stack_info.setWordWrap(True)
    f.addRow("Apply", win.cmb_disp_mode)
    f.addRow(win.lbl_stack_info)

    # Compare
    g_cmp = QGroupBox("Compare")
    v = _vbox(g_cmp)
    win.btn_prop_load_meas = QPushButton("Load measured…")
    win.btn_prop_load_meas.setObjectName("accentcompact")
    win.btn_prop_load_meas.setToolTip(
        "A trace (.dwc or prepared .npz) measured after the stack. A .dwc goes "
        "through the Prepare "
        "page's settings; either kind is put onto the pulse's grid.")
    win.lbl_prop_meas = ElidedLabel("")
    win.lbl_prop_meas.setObjectName("dim")
    win.lbl_verdict = QLabel("")
    win.lbl_verdict.setObjectName("readout_sm")
    win.lbl_verdict.setToolTip(
        "SHG FROG cannot tell E(t) from E*(−t). Through material the two "
        "differ: whichever matches the measurement is the pulse's direction. "
        "The scan varies the stack's dispersion by a factor s: a minimum near "
        "+1 confirms the retrieval, near −1 means it is time-reversed.")
    win.btn_prop_adv = QPushButton("Scan…")
    win.btn_prop_adv.setObjectName("compact")
    win.btn_prop_adv.setToolTip("Direction-scan settings")
    v.addLayout(_row(win.btn_prop_load_meas, win.btn_prop_adv))
    v.addWidget(win.lbl_prop_meas)
    v.addWidget(win.lbl_verdict)

    # Advanced (pop-up)
    win.sp_scan_range = DoubleSpinBox()
    win.sp_scan_range.setRange(0.5, 20.0)
    win.sp_scan_range.setDecimals(1)
    win.sp_scan_range.setSingleStep(0.5)
    win.sp_scan_range.setValue(2.5)
    win.sp_scan_range.setPrefix("± ")
    win.sp_scan_points = SpinBox()
    win.sp_scan_points.setRange(11, 401)
    win.sp_scan_points.setSingleStep(10)
    win.sp_scan_points.setValue(61)
    win.dlg_prop_adv = AdvancedDialog("Propagator — advanced", [
        ("Scan range", win.sp_scan_range,
         "The direction scan covers s = −range … +range times the stack's "
         "dispersion."),
        ("Scan points", win.sp_scan_points,
         "SHG traces computed for the scan. More is smoother but slower on "
         "large grids."),
    ], win)

    win.propagator_canvas = PropagatorCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    win.propagator_column = _column(g_pulse, g_stack, g_disp, g_cmp)
    h.addWidget(win.propagator_column)
    h.addWidget(win.propagator_canvas, 1)
    return page


def build_statusbar(win) -> None:
    sb = QStatusBar()
    sb.setSizeGripEnabled(False)
    win.setStatusBar(sb)
    win.lbl_engine = QLabel("Julia: off")
    win.lbl_engine.setObjectName("satok")
    win.engine_lamp = StatusLamp()
    sb.addPermanentWidget(win.lbl_engine)
    sb.addPermanentWidget(win.engine_lamp)


def build_ui(win) -> None:
    build_toolbar(win)
    win.stack = QStackedWidget()
    win.stack.addWidget(build_prep_page(win))
    win.stack.addWidget(build_recon_page(win))
    win.stack.addWidget(build_analysis_page(win))
    win.stack.addWidget(build_propagator_page(win))
    win.setCentralWidget(win.stack)
    build_statusbar(win)
    win.btn_theme.setIcon(QIcon())
