"""Create juliacall's Julia environment (julia/env) and precompile everything.

Run once after cloning, and again after changing julia/TadpoleRetrieval's
dependencies:

    .venv\\Scripts\\python scripts\\setup_julia.py

juliapkg reads tadpole/juliapkg.json (it scans sys.path, and the repo root is
on it), adds PythonCall plus a `dev` of julia/TadpoleRetrieval to julia/env,
and instantiates it with the Julia named by tadpole.paths.julia_executable().
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tadpole.paths import julia_environment, julia_executable  # noqa: E402


def main() -> int:
    exe = julia_executable()
    if not exe:
        print("No julia found. Install Julia (juliaup) or set TADPOLE_JULIA.")
        return 1
    print(f"Julia:        {exe}")
    env = julia_environment()
    print(f"Environment:  {env['PYTHON_JULIAPKG_PROJECT']}")
    os.environ.update(env)
    t0 = time.perf_counter()
    from juliacall import Main as jl
    jl.seval("using TadpoleRetrieval")
    print(f"Julia {jl.seval('string(VERSION)')} ready, TadpoleRetrieval loaded "
          f"in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
