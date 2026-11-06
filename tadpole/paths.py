"""Where Tadpole's files live — the one module that resolves a path.

Same rule as Lillypad: this is the ONLY place that derives a location from
`__file__`.

  resource_path()  BUNDLED, read-only: icons, the stylesheet, and (frozen) the
                   Julia bundle (runtime, depot, env).
  app_dir()        BESIDE THE PROGRAM, writable: settings.json.

The Julia locations are here too, because the worker process has to point
juliacall at them BEFORE `import juliacall`, and a frozen build ships them in a
different place than a source checkout.

No Qt, no numpy, no juliacall: this module must import in a bare interpreter.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts) -> Path:
    """A bundled resource, from source (repo root) or frozen (_MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", _PROJECT_ROOT))
    return base.joinpath(*parts)


def app_dir() -> Path:
    """Folder the program lives in: next to the .exe when frozen, else the
    repository root. Writable; user data belongs here."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _PROJECT_ROOT


SETTINGS_PATH = app_dir() / "settings.json"

# ── Julia ────────────────────────────────────────────────────────────────────
# Two ways to find Julia:
#
#   source checkout  the installed Julia (PATH / $TADPOLE_JULIA), the package in
#                    julia/TadpoleRetrieval, and juliacall's environment julia/env
#                    (PythonCall + a dev of the package, declared by
#                    tadpole/juliapkg.json, built by scripts/setup_julia.py)
#   bundle           a self-contained runtime + depot + env built by
#                    scripts/build_julia_bundle.py; always used when frozen
JULIA_PACKAGE_DIR = _PROJECT_ROOT / "julia" / "TadpoleRetrieval"
JULIA_ENV_DIR = _PROJECT_ROOT / "julia" / "env"
JULIA_SYSIMAGE = _PROJECT_ROOT / "julia" / "sysimage" / (
    "tadpole.dll" if os.name == "nt" else "tadpole.so")


def julia_executable() -> str | None:
    """The julia binary for a source checkout: $TADPOLE_JULIA, else `julia`
    on PATH."""
    return os.environ.get("TADPOLE_JULIA") or shutil.which("julia")


def julia_bundle_dir() -> Path | None:
    """The self-contained Julia bundle (scripts/build_julia_bundle.py), if in
    use: always when frozen, or from source when $TADPOLE_JULIA_BUNDLE names
    one (to test a bundle without building the exe)."""
    env = os.environ.get("TADPOLE_JULIA_BUNDLE")
    if env:
        return Path(env).resolve()
    if is_frozen():
        b = resource_path("julia_bundle")
        if b.is_dir():
            return b
    return None


def _bundle_environment(bundle: Path) -> dict[str, str]:
    """Point juliacall straight at the bundle's julia.exe and environment.

    With both PYTHON_JULIACALL_EXE and _PROJECT set, juliacall skips juliapkg:
    nothing is resolved or downloaded. The depot path lists a WRITABLE user
    depot first (anything Julia has to (re)compile lands there) and the
    read-only bundled depot second.
    """
    user = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Tadpole" / "julia_depot"
    exe = bundle / "runtime" / "bin" / ("julia.exe" if os.name == "nt" else "julia")
    env = {
        "PYTHON_JULIACALL_EXE": str(exe),
        "PYTHON_JULIACALL_PROJECT": str(bundle / "env"),
        "JULIA_DEPOT_PATH": os.pathsep.join([str(user), str(bundle / "depot")]),
        "JULIA_PKG_OFFLINE": "true",
        "JULIA_CONDAPKG_BACKEND": "Null",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "PYTHON_JULIACALL_THREADS": "1",
    }
    sysimage = bundle / "sysimage" / JULIA_SYSIMAGE.name
    if sysimage.exists():
        env["PYTHON_JULIACALL_SYSIMAGE"] = str(sysimage)
    return env


def julia_environment() -> dict[str, str]:
    """Environment variables the worker sets before importing juliacall."""
    bundle = julia_bundle_dir()
    if bundle is not None:
        return _bundle_environment(bundle)
    env = {
        "PYTHON_JULIAPKG_PROJECT": str(JULIA_ENV_DIR),
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "PYTHON_JULIACALL_THREADS": "1",
    }
    exe = julia_executable()
    if exe:
        # Always name a Julia, so juliapkg never downloads one of its own.
        env["PYTHON_JULIAPKG_EXE"] = exe
    if JULIA_SYSIMAGE.exists():
        env["PYTHON_JULIACALL_SYSIMAGE"] = str(JULIA_SYSIMAGE)
    return env
