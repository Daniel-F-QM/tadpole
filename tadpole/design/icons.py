"""Icon files on disk, and the application icon.

Same conventions as Lillypad:

  * A two-state PAIR shows the view a click GIVES you (SHOW/HIDE).
  * An action with no second state has ONE mark in two files, differing only in
    the theme accent it is drawn in (RESCALE, PICTURE, REFRESH).

Every path is resolved through tadpole.paths.resource_path, so the constants
work from source and from the frozen bundle.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon

from tadpole.paths import resource_path

# No Tadpole artwork yet: the window icon falls back to Qt's default until
# icons/Tadpole.ico exists.
ICON_ICO_PATH = resource_path("icons", "Tadpole.ico")
ICON_PATH     = resource_path("icons", "Tadpole.png")
SUN_ICON      = resource_path("icons", "sun.png")
MOON_ICON     = resource_path("icons", "moon.png")
PICTURE_ICON = {"dark":  resource_path("icons", "save_picture_dark.png"),
                "light": resource_path("icons", "save_picture_light.png")}
RESCALE_ICON = {"dark":  resource_path("icons", "rescale_dark.png"),
                "light": resource_path("icons", "rescale_light.png")}
REFRESH_ICON = {"dark":  resource_path("icons", "refresh_dark.png"),
                "light": resource_path("icons", "refresh_light.png")}
SHOW_ICON = {"dark":  resource_path("icons", "show_dark.png"),
             "light": resource_path("icons", "show_light.png")}
HIDE_ICON = {"dark":  resource_path("icons", "hide_dark.png"),
             "light": resource_path("icons", "hide_light.png")}
SYMMETRY_ICON = resource_path("icons", "symmetry.png")

_APP_ICON = None


def app_icon() -> QIcon:
    """The application icon, built once (a QIcon needs a QApplication)."""
    global _APP_ICON
    if _APP_ICON is None:
        src = ICON_ICO_PATH if ICON_ICO_PATH.exists() else ICON_PATH
        _APP_ICON = QIcon(str(src)) if src.exists() else QIcon()
        for px in (16, 20, 24, 32, 40, 48, 64, 256):
            _APP_ICON.pixmap(px, px)
    return _APP_ICON
