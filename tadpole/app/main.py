"""Starting the program — Lillypad's startup order:

  1. the AppUserModelID, before any window exists (own taskbar icon);
  2. the QApplication, its name, icon and base font, "." decimals;
  3. **the theme, before the first widget** (PALETTE is sampled at
     construction time);
  4. the window, maximized; it starts the Julia worker in the background.
"""

import multiprocessing
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from tadpole.design.icons import app_icon
from tadpole.design.theme import install_theme, startup_theme
from tadpole.gui.widgets import use_dot_decimals
from tadpole.gui.window.main_window import TadpoleWindow
from tadpole.settings import load_settings


def main():
    # The Julia worker is a "spawn" child process; a frozen exe must not run
    # the GUI again in it.
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        from tadpole.app.selftest import run
        i = sys.argv.index("--selftest")
        sys.exit(run(sys.argv[i + 1] if len(sys.argv) > i + 1 else None))
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "AlcortaFleischmann.Tadpole")
    except (AttributeError, OSError):
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("Tadpole")
    app.setWindowIcon(app_icon())
    app.setFont(QFont("Segoe UI", 10, QFont.Normal))
    use_dot_decimals()
    theme = startup_theme(load_settings())
    install_theme(app, theme)

    win = TadpoleWindow(theme)
    win.show_restored()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".dwc"):
        win.prep.open_path(sys.argv[1])
    sys.exit(app.exec())
