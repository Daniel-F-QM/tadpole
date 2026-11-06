"""Building the window's widgets — free functions, like Lillypad's build_ui.

Layout (Lillypad's): a toolbar, then per page a fixed-width scrolled control
column of QGroupBoxes on the left and the figure on the right. Two pages,
Prepare and Reconstruct, in a QStackedWidget switched by the toolbar's mode
buttons. Widgets are attached to `win` by name; the controllers wire them.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget,
                               QStatusBar, QToolBar, QVBoxLayout, QWidget)

from tadpole.engine.protocol import ALGORITHMS
from tadpole.gui.canvas.prep_canvas import PrepCanvas
from tadpole.gui.canvas.recon_canvas import ReconCanvas
from tadpole.gui.widgets import (ComboBox, DoubleSpinBox, OptionalDoubleSpinBox,
                                 SpinBox, StatusLamp, _hline, dim_label)

PANEL_W = 312

ALGORITHM_NOTES = {
    "pcgpa": "Principal-component generalized projections. Fast and robust; "
             "needs the square N×N grid.",
    "gpa": "Generalized projections with an exact line search on the "
           "mathematical-form constraint. Slower per iteration than PCGPA.",
    "ptychographic": "One measured spectrum per update; works on the measured "
                     "delays and with missing data. Not implemented yet.",
}


def _scroll_column(*groups) -> QScrollArea:
    col = QWidget()
    col.setMinimumWidth(PANEL_W - 32)
    col.setMaximumWidth(PANEL_W - 12)
    lay = QVBoxLayout(col)
    lay.setContentsMargins(0, 0, 6, 0)
    lay.setSpacing(8)
    for g in groups:
        lay.addWidget(g)
    lay.addStretch()
    sa = QScrollArea()
    sa.setWidget(col)
    sa.setWidgetResizable(True)
    sa.setFixedWidth(PANEL_W)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


def _form(box: QGroupBox) -> QFormLayout:
    f = QFormLayout(box)
    f.setContentsMargins(8, 14, 8, 8)
    f.setHorizontalSpacing(10)
    f.setVerticalSpacing(6)
    f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return f


def build_toolbar(win) -> None:
    tb = QToolBar()
    tb.setMovable(False)
    tb.setIconSize(QSize(18, 18))
    win.addToolBar(tb)
    win.btn_mode_prep = QPushButton("Prepare")
    win.btn_mode_recon = QPushButton("Reconstruct")
    win.mode_group = QButtonGroup(win)
    for i, b in enumerate((win.btn_mode_prep, win.btn_mode_recon)):
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
    v = QVBoxLayout(g_file)
    v.setContentsMargins(8, 14, 8, 8)
    win.lbl_file = QLabel("No trace loaded")
    win.lbl_file.setWordWrap(True)
    win.lbl_file_info = dim_label("")
    v.addWidget(win.lbl_file)
    v.addWidget(win.lbl_file_info)

    # Background
    g_bg = QGroupBox("Background")
    f = _form(g_bg)
    win.cmb_bg = ComboBox()
    win.cmb_bg.addItem("None", "none")
    win.cmb_bg.addItem("Constant (edge mean)", "constant")
    win.cmb_bg.addItem("Spectral (edge spectrum)", "spectral")
    win.sp_edge_cols = SpinBox(); win.sp_edge_cols.setRange(0, 200)
    win.sp_edge_rows = SpinBox(); win.sp_edge_rows.setRange(0, 500)
    f.addRow("Mode", win.cmb_bg)
    f.addRow("Edge delays", win.sp_edge_cols)
    f.addRow("Edge λ rows", win.sp_edge_rows)
    f.addRow(dim_label("Edge delays: columns at each end of the delay axis used "
                       "as background. Edge λ rows: additionally removes a "
                       "per-delay offset. Negative pixels are kept until the "
                       "trace is binned onto the grid, so noise averages out."))

    # Crop
    g_crop = QGroupBox("Crop")
    f = _form(g_crop)
    win.sp_wl_min = OptionalDoubleSpinBox(0.0, 10000.0, "full", 1, 1.0)
    win.sp_wl_max = OptionalDoubleSpinBox(0.0, 10000.0, "full", 1, 1.0)
    win.sp_d_min = OptionalDoubleSpinBox(-100000.0, 100000.0, "full", 1, 5.0)
    win.sp_d_max = OptionalDoubleSpinBox(-100000.0, 100000.0, "full", 1, 5.0)
    for sp in (win.sp_wl_min, win.sp_wl_max):
        sp.setSuffix(" nm")
    for sp in (win.sp_d_min, win.sp_d_max):
        sp.setSuffix(" fs")
    f.addRow("λ min", win.sp_wl_min)
    f.addRow("λ max", win.sp_wl_max)
    f.addRow("Delay min", win.sp_d_min)
    f.addRow("Delay max", win.sp_d_max)

    # Delay zero
    g_zero = QGroupBox("Delay zero")
    f = _form(g_zero)
    win.cmb_zero = ComboBox()
    win.cmb_zero.addItem("Centroid of marginal", "centroid")
    win.cmb_zero.addItem("Peak of marginal", "peak")
    win.cmb_zero.addItem("As in file (centre column)", "none")
    win.sp_zero_off = DoubleSpinBox(); win.sp_zero_off.setRange(-10000, 10000)
    win.sp_zero_off.setDecimals(2); win.sp_zero_off.setSuffix(" fs")
    win.chk_sym = QCheckBox("Symmetrize in delay (SHG)")
    f.addRow("Zero from", win.cmb_zero)
    f.addRow("Extra shift", win.sp_zero_off)
    f.addRow(win.chk_sym)

    # Fourier grid
    g_grid = QGroupBox("Fourier grid")
    f = _form(g_grid)
    win.cmb_n = ComboBox()
    for n in (64, 128, 256, 512, 1024):
        win.cmb_n.addItem(str(n), n)
    win.sp_dt = DoubleSpinBox(); win.sp_dt.setRange(0.0, 1000.0)
    win.sp_dt.setDecimals(3); win.sp_dt.setSingleStep(0.1)
    win.sp_dt.setSpecialValueText("auto"); win.sp_dt.setSuffix(" fs")
    win.lbl_grid = QLabel("")
    win.lbl_grid.setObjectName("readout_sm")
    win.lbl_grid.setWordWrap(True)
    win.lbl_warn = QLabel("")
    win.lbl_warn.setObjectName("satwarn")
    win.lbl_warn.setWordWrap(True)
    f.addRow("N", win.cmb_n)
    f.addRow("dt", win.sp_dt)
    f.addRow(win.lbl_grid)
    f.addRow(win.lbl_warn)
    f.addRow(dim_label("dω = 2π/(N·dt). Auto dt leaves the same headroom on the "
                       "delay and the frequency window."))

    # Output
    g_out = QGroupBox("Output")
    v = QVBoxLayout(g_out)
    v.setContentsMargins(8, 14, 8, 8)
    win.btn_send = QPushButton("Send to Reconstructor")
    win.btn_send.setObjectName("accent")
    win.btn_save_prep = QPushButton("Save prepared (.npz)")
    win.btn_save_prep.setObjectName("compact")
    v.addWidget(win.btn_send)
    v.addWidget(win.btn_save_prep)

    win.prep_canvas = PrepCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    h.addWidget(_scroll_column(g_file, g_bg, g_crop, g_zero, g_grid, g_out))
    h.addWidget(win.prep_canvas, 1)
    return page


def build_recon_page(win) -> QWidget:
    g_in = QGroupBox("Input")
    v = QVBoxLayout(g_in)
    v.setContentsMargins(8, 14, 8, 8)
    win.lbl_recon_input = QLabel("Nothing prepared yet")
    win.lbl_recon_input.setWordWrap(True)
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
    win.lbl_alg_note = dim_label("")
    win.cmb_init = ComboBox()
    win.cmb_init.addItem("Autocorrelation", "autocorrelation")
    win.cmb_init.addItem("Gaussian", "gaussian")
    win.cmb_init.addItem("Random", "random")
    win.sp_seed = SpinBox(); win.sp_seed.setRange(0, 999999)
    win.sp_alpha_min = DoubleSpinBox(); win.sp_alpha_min.setRange(0.0, 2.0)
    win.sp_alpha_max = DoubleSpinBox(); win.sp_alpha_max.setRange(0.0, 2.0)
    for sp in (win.sp_alpha_min, win.sp_alpha_max):
        sp.setDecimals(2); sp.setSingleStep(0.05)
    win.sp_gamma = DoubleSpinBox(); win.sp_gamma.setRange(0.0, 1.0)
    win.sp_gamma.setDecimals(7); win.sp_gamma.setSingleStep(1e-6)
    win.sp_gamma.setSpecialValueText("off")
    f.addRow(win.cmb_alg)
    f.addRow(win.lbl_alg_note)
    f.addRow("Initial guess", win.cmb_init)
    f.addRow("Seed", win.sp_seed)
    win.ptycho_rows = []
    for label, w in (("α min", win.sp_alpha_min), ("α max", win.sp_alpha_max),
                     ("γ threshold", win.sp_gamma)):
        lbl = QLabel(label)
        f.addRow(lbl, w)
        win.ptycho_rows.append((lbl, w))

    g_stop = QGroupBox("Stopping")
    f = _form(g_stop)
    win.sp_max_iter = SpinBox(); win.sp_max_iter.setRange(1, 1000000)
    win.sp_max_iter.setSingleStep(100)
    win.sp_tol_exp = SpinBox(); win.sp_tol_exp.setRange(-12, -1)
    win.sp_tol_exp.setPrefix("G < 10^")
    f.addRow("Max iterations", win.sp_max_iter)
    f.addRow("Tolerance", win.sp_tol_exp)

    g_run = QGroupBox("Run")
    v = QVBoxLayout(g_run)
    v.setContentsMargins(8, 14, 8, 8)
    row = QHBoxLayout()
    win.btn_start = QPushButton("Start")
    win.btn_start.setObjectName("accent")
    win.btn_stop = QPushButton("Stop")
    win.btn_stop.setObjectName("danger")
    row.addWidget(win.btn_start)
    row.addWidget(win.btn_stop)
    v.addLayout(row)
    win.progress = QProgressBar()
    win.progress.setFixedHeight(18)
    win.progress.setRange(0, 100)
    win.progress.setValue(0)
    v.addWidget(win.progress)
    v.addWidget(_hline())
    grid = QFormLayout()
    grid.setVerticalSpacing(2)
    win.lbl_G = QLabel("—"); win.lbl_G.setObjectName("readout")
    win.lbl_Gp = QLabel("—"); win.lbl_Gp.setObjectName("readout")
    win.lbl_iter = QLabel("—"); win.lbl_iter.setObjectName("readout_sm")
    g_lbl = QLabel("G"); g_lbl.setObjectName("hdr")
    gp_lbl = QLabel("G′"); gp_lbl.setObjectName("hdr")
    it_lbl = QLabel("Iteration"); it_lbl.setObjectName("hdr")
    grid.addRow(g_lbl, win.lbl_G)
    grid.addRow(gp_lbl, win.lbl_Gp)
    grid.addRow(it_lbl, win.lbl_iter)
    v.addLayout(grid)
    v.addWidget(dim_label("G: rms trace error (Trebino). G′: the same residual "
                          "normalised by the measured trace's norm, so it does "
                          "not depend on grid size."))

    g_res = QGroupBox("Result")
    v = QVBoxLayout(g_res)
    v.setContentsMargins(8, 14, 8, 8)
    win.lbl_result = dim_label("No result yet")
    win.btn_save_result = QPushButton("Save result (.npz / .csv)…")
    win.btn_save_result.setObjectName("compact")
    v.addWidget(win.lbl_result)
    v.addWidget(win.btn_save_result)

    win.recon_canvas = ReconCanvas()
    page = QWidget()
    h = QHBoxLayout(page)
    h.setContentsMargins(8, 8, 8, 8)
    h.setSpacing(8)
    h.addWidget(_scroll_column(g_in, g_alg, g_stop, g_run, g_res))
    h.addWidget(win.recon_canvas, 1)
    return page


def build_statusbar(win) -> None:
    sb = QStatusBar()
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
    win.setCentralWidget(win.stack)
    build_statusbar(win)
    win.btn_theme.setIcon(QIcon())
