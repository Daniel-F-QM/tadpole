"""Build the self-contained Julia bundle that the frozen exe ships.

    .venv\\Scripts\\python scripts\\build_julia_bundle.py

Produces build/julia_bundle/:

    runtime/   a copy of the Julia installation (bin, lib, share minus docs)
    depot/     a private package depot: TadpoleRetrieval under depot/dev,
               PythonCall, FFTW…, all precompiled
    env/       the environment juliacall activates (Project + Manifest with
               RELATIVE paths, so the folder can move)

At run time tadpole.paths points juliacall straight at runtime/bin/julia.exe
and env/ (PYTHON_JULIACALL_EXE + PYTHON_JULIACALL_PROJECT), which bypasses
juliapkg entirely: nothing is resolved or downloaded on the user's machine.

TadpoleRetrieval sits INSIDE the depot on purpose: Julia ≥ 1.11 records
depot-relative paths for code in a depot, so the precompile caches stay valid
after the bundle is copied somewhere else. A package dev'd from outside any
depot would be recompiled on every new machine.

Lazy artifacts (MKL, ~0.4 GB, pulled in by FFTW) are installed explicitly:
the sysimage initialises MKL_jll at start-up, and a bundle without the
artifact would try to download it on the user's machine — and fail offline.
`--artifacts-only` adds them to an existing bundle without rebuilding it.

Test the bundle without PyInstaller:

    $env:TADPOLE_JULIA_BUNDLE = "build\\julia_bundle"; .venv\\Scripts\\python main.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "julia_bundle"
EXE = "julia.exe" if os.name == "nt" else "julia"


def julia_home() -> Path:
    exe = os.environ.get("TADPOLE_JULIA") or shutil.which("julia")
    if not exe:
        raise SystemExit("julia not found")
    home = subprocess.check_output([exe, "--startup-file=no", "-e",
                                    "print(dirname(Sys.BINDIR))"], text=True)
    return Path(home.strip())


def pythoncall_version() -> str:
    """The exact PythonCall version juliacall requires — read from its
    juliapkg.json WITHOUT importing juliacall (importing it boots Julia)."""
    import importlib.util
    origin = importlib.util.find_spec("juliacall").origin
    spec = json.loads((Path(origin).parent / "juliapkg.json").read_text())
    return spec["packages"]["PythonCall"]["version"].lstrip("=")


# Install every artifact (lazy ones included) of every package in the env.
ARTIFACTS_JL = """
using Pkg, TOML
import Pkg.Artifacts: ensure_artifact_installed
for (uuid, dep) in Pkg.dependencies()
    dep.source === nothing && continue
    toml = joinpath(dep.source, "Artifacts.toml")
    isfile(toml) || continue
    for name in keys(TOML.parsefile(toml))
        ensure_artifact_installed(name, toml)
    end
end
println("artifacts ok")
"""


def run_julia(script: str) -> None:
    depot = OUT / "depot"
    env = dict(os.environ, JULIA_DEPOT_PATH=str(depot),
               JULIA_CONDAPKG_BACKEND="Null", JULIA_PKG_PRECOMPILE_AUTO="0")
    subprocess.check_call([str(OUT / "runtime" / "bin" / EXE), "--startup-file=no",
                           "--project=.", "-e", script], cwd=OUT / "env", env=env)


def report_size() -> None:
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Bundle ready: {OUT}  ({size / 2**20:.0f} MB)")


def main() -> int:
    if "--artifacts-only" in sys.argv:
        if not OUT.is_dir():
            raise SystemExit(f"no bundle at {OUT}")
        run_julia(ARTIFACTS_JL)
        report_size()
        return 0
    home = julia_home()
    print(f"Julia installation: {home}")
    if OUT.exists():
        shutil.rmtree(OUT)
    runtime, depot, env = OUT / "runtime", OUT / "depot", OUT / "env"

    print("Copying runtime…")
    shutil.copytree(home, runtime,
                    ignore=shutil.ignore_patterns("doc", "include", "*.pdb"))

    print("Copying TadpoleRetrieval into the depot…")
    pkg_src = ROOT / "julia" / "TadpoleRetrieval"
    shutil.copytree(pkg_src, depot / "dev" / "TadpoleRetrieval",
                    ignore=shutil.ignore_patterns("test"))

    env.mkdir(parents=True)
    pc = pythoncall_version()
    script = f"""
    using Pkg
    Pkg.develop(path="../depot/dev/TadpoleRetrieval")
    Pkg.add(name="PythonCall", version="{pc}")
    Pkg.compat("PythonCall", "={pc}")
    Pkg.instantiate()
    Pkg.precompile()
    using TadpoleRetrieval, PythonCall
    println("bundle env ok")
    """
    print(f"Instantiating env (PythonCall {pc}) — downloads packages once…")
    run_julia(script)
    print("Installing lazy artifacts…")
    run_julia(ARTIFACTS_JL)

    # Not needed at run time: the registry (no resolving happens there) and
    # download logs. Packages, compiled caches, artifacts and scratch spaces stay.
    for sub in ("registries", "logs", "clones"):
        shutil.rmtree(depot / sub, ignore_errors=True)

    report_size()
    return 0


if __name__ == "__main__":
    sys.exit(main())
