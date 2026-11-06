"""The Julia worker process — the ONLY module that imports juliacall.

Runs in its own process (multiprocessing "spawn"), so Julia's runtime never
shares a process with Qt: no GIL contention with the GUI thread, no clash of
signal handlers, and a hung or crashed retrieval can be killed outright and
the worker restarted.

Life cycle:
  1. point juliacall at our Julia + environment (tadpole.paths), import it,
     `using TadpoleRetrieval` (precompiled, so this is seconds, not minutes)
  2. send Ready
  3. serve Start messages one at a time; during a run the progress callback
     forwards Snapshots and checks for a Stop between iterations
"""

from __future__ import annotations

import os
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from tadpole.engine.protocol import (Done, Error, Log, Quit, Ready, Snapshot,
                                     Start, Stop)


# The Python modules PythonCall imports while it initialises. In a frozen exe
# PyInstaller only bundles what it saw imported, and a missing one kills the
# worker inside Julia's start-up with an opaque InitError, so check them first.
PYTHONCALL_IMPORTS = ("sys", "io", "inspect", "traceback", "numbers",
                      "collections", "collections.abc", "juliacall", "numpy")


def _check_python_imports() -> None:
    import importlib
    missing = []
    for name in PYTHONCALL_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:          # noqa: BLE001 - report every failure
            missing.append(f"{name}: {exc!r}")
    if missing:
        raise ImportError("modules PythonCall needs are missing: " + "; ".join(missing))


def _boot_julia():
    from tadpole.paths import julia_environment
    os.environ.update(julia_environment())
    _check_python_imports()
    from juliacall import Main as jl        # noqa: E402  (env must be set first)
    jl.seval("using TadpoleRetrieval, PythonCall")
    return jl


def _to_numpy(x):
    return np.array(x)


def _result_to_python(res) -> dict:
    out = {}
    for key in ("E", "trace", "errors", "errors_p", "t", "w"):
        out[key] = _to_numpy(res[key])
    for key in ("G", "Gp"):
        out[key] = float(res[key])
    out["iterations"] = int(res["iterations"])
    out["stopped"] = str(res["stopped"])
    return out


def run_job(jl, job: Start, req_q, resp_q, helpers) -> None:
    """Run one Start; every outcome ends in exactly one Done or Error."""
    to_dict, wrap_cb = helpers
    stop_requested = False

    def on_progress(it, G, Gp, T, E):
        nonlocal stop_requested
        # Drain the request queue: a Stop for this job ends the run, anything
        # else (a second Start) is put back for after it.
        pending = []
        while True:
            try:
                msg = req_q.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, Stop) and msg.job_id == job.job_id:
                stop_requested = True
            elif isinstance(msg, Quit):
                stop_requested = True
                pending.append(msg)
            elif not isinstance(msg, Stop):
                pending.append(msg)
        for msg in pending:
            req_q.put(msg)
        resp_q.put(Snapshot(job.job_id, int(it), float(G), float(Gp),
                            _to_numpy(T), _to_numpy(E)))
        return stop_requested

    t0 = time.perf_counter()
    try:
        kwargs = {"options": to_dict(dict(job.options)), "callback": wrap_cb(on_progress)}
        if job.mask is not None:
            kwargs["mask"] = np.ascontiguousarray(job.mask, dtype=bool)
        if job.delays_fs is not None:
            kwargs["delays"] = np.ascontiguousarray(job.delays_fs, dtype=float)
        res = jl.TadpoleRetrieval.run_retrieval(
            job.algorithm, np.ascontiguousarray(job.trace, dtype=float),
            float(job.dt_fs), **kwargs)
        out = _result_to_python(res)
        out["algorithm"] = job.algorithm
        out["elapsed_s"] = time.perf_counter() - t0
        resp_q.put(Done(job.job_id, out))
    except Exception as exc:                 # JuliaError included
        resp_q.put(Error(job.job_id, short_error(exc)))


def short_error(exc) -> str:
    """A Julia error without its stack trace: the message is what the operator
    needs; the trace is noise in a status bar."""
    msg = str(exc).split("Stacktrace:")[0].strip()
    return msg or type(exc).__name__


def _warm_up(jl, to_dict, wrap_cb) -> None:
    """Compile the numpy-array (PyArray) method instances before Ready.

    The package's precompile workload covers Julia-native arrays only; the
    first call with arrays coming from Python would otherwise spend several
    seconds compiling while the operator watches a frozen progress bar.
    """
    d = jl.TadpoleRetrieval.simulate(16, 4.0)
    T = np.ascontiguousarray(np.array(d["trace"]))
    mask = np.ones(T.shape, dtype=bool)
    for alg in ("pcgpa", "gpa"):
        for m in (None, mask):
            kw = {} if m is None else {"mask": m}
            jl.TadpoleRetrieval.run_retrieval(
                alg, T, 4.0, **kw,
                options=to_dict({"max_iter": 2, "min_snapshot_s": 0.0}),
                callback=wrap_cb(lambda *a: False))


def worker_log_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Tadpole" / "julia_worker.log"


def _redirect_output_if_headless() -> None:
    """A windowed (frozen) exe has no console: the worker's stdout/stderr, and
    with them Julia's messages and any traceback, would vanish. Send them to
    a log file instead."""
    if sys.stderr is not None and sys.stdout is not None:
        return
    path = worker_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, "w", encoding="utf-8", buffering=1)
    except OSError:
        return
    sys.stdout = sys.stderr = f
    # Julia's C runtime writes to the OS-level handles, not to sys.stderr.
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        if os.name == "nt":
            import ctypes
            import msvcrt
            h = msvcrt.get_osfhandle(f.fileno())
            for std in (-11, -12):              # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
                ctypes.windll.kernel32.SetStdHandle(std, h)
    except (OSError, AttributeError):
        pass


def main(req_q, resp_q) -> None:
    """Process entry point (picklable top-level function for spawn)."""
    _redirect_output_if_headless()
    t0 = time.perf_counter()
    try:
        jl = _boot_julia()
        # A Python dict arrives as PyDict{Any,Any} (keys/values already
        # converted to native Julia values); the API wants Dict{String,Any}.
        to_dict = jl.seval("d -> Dict{String,Any}(string(k) => v for (k, v) in d)")
        wrap_cb = jl.seval("f -> (it, G, Gp, T, E) -> pytruth(f(it, G, Gp, T, E))")
        version = str(jl.seval("string(VERSION)"))
        _warm_up(jl, to_dict, wrap_cb)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        resp_q.put(Error(None, "Julia failed to start:\n" + tb))
        return
    resp_q.put(Ready(version, time.perf_counter() - t0))

    while True:
        msg = req_q.get()
        if isinstance(msg, Quit):
            break
        if isinstance(msg, Start):
            run_job(jl, msg, req_q, resp_q, (to_dict, wrap_cb))
        elif isinstance(msg, Stop):
            continue                          # nothing running: nothing to stop
        else:
            resp_q.put(Log(f"worker ignored unknown message {type(msg).__name__}"))
