"""Reading and writing settings.json (same mechanism as Lillypad).

Display preferences and preparation/retrieval defaults only. A plain JSON file
beside the program, so it can be read, edited and deleted (= factory reset).

No Qt here — main() needs the theme out of this file before the QApplication is
styled.
"""

from __future__ import annotations

import copy
import json
import os

from tadpole.paths import SETTINGS_PATH

SETTINGS_VERSION = 1

# What a fresh install, or a deleted settings.json, starts with. A key missing
# here falls back to the value its widget is built with.
DEFAULT_SETTINGS = {
    "theme": "dark",
    "window": {"x": 0, "y": 0, "w": 1280, "h": 800, "maximized": True},
    "cmap": "jet",
    "cmap_reversed": False,
    "log_scale": False,
    "last_dir": "",
    # preparation
    "bg_mode": "spectral",
    "bg_edge_cols": 5,
    "bg_edge_rows": 0,
    "grid_n": 128,
    "grid_dt_fs": 0.0,         # 0 → automatic (see prep.grid.auto_dt)
    "wl_min_nm": None,
    "wl_max_nm": None,
    "zero_mode": "centroid",
    "symmetrize": False,
    # reconstruction
    "algorithm": "pcgpa",
    "max_iter": 500,
    "tol_exp": -5,             # stop at G < 10^tol_exp
    "init": "autocorrelation",
    "seed": 0,
    "alpha_min": 0.1,
    "alpha_max": 0.5,
    "gamma": 0.0,
}

_SETTINGS = None


def load_settings() -> dict:
    """DEFAULT_SETTINGS overlaid with settings.json, read once per process.

    A missing or corrupt file means the defaults alone: a preferences file that
    lost a brace must never stop the program from starting.
    """
    global _SETTINGS
    if _SETTINGS is None:
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, UnicodeDecodeError):
            data = {}
        merged = copy.deepcopy(DEFAULT_SETTINGS)
        merged.update(data if isinstance(data, dict) else {})
        _SETTINGS = merged
    return _SETTINGS


def save_settings(data) -> bool:
    """Write settings.json atomically (temp file + os.replace). True if it
    landed; failure is silent, like Lillypad's."""
    tmp = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=float)
            f.write("\n")
        os.replace(tmp, SETTINGS_PATH)
        return True
    except (OSError, TypeError, ValueError):
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def as_bool(v) -> bool:
    """Strict bool for restore: bool("false") is True, so anything that is not
    a real JSON boolean raises and that key falls back to its default."""
    if isinstance(v, bool):
        return v
    raise TypeError(v)


def _reset_cache() -> None:
    """Drop the process-lifetime cache. For tests only."""
    global _SETTINGS
    _SETTINGS = None
