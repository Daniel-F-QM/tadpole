"""Messages between the GUI process and the Julia worker process.

Plain picklable dataclasses carrying numpy arrays — nothing Julia-typed ever
crosses the process boundary.

GUI → worker:  Start, Stop, Quit
worker → GUI:  Ready, Snapshot, Done, Error, Log
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ALGORITHMS = {
    "pcgpa": "PCGPA (Kane 1999)",
    "gpa": "GPA (DeLong & Trebino 1994)",
    "ptychographic": "Ptychographic (Sidorenko 2016) — not implemented",
}

# Algorithms listed in the GUI whose Julia side is not written yet: the GUI
# will not start them. Remove a name once its retrieve_* function works.
NOT_IMPLEMENTED = {"ptychographic"}


@dataclass
class Start:
    """Run one retrieval.

    `trace` is I[ω, τ] (rows frequency offsets on the N-point grid with time
    step `dt_fs`). For "ptychographic", `delays_fs` lists the measured delay of
    each column and the trace may be N×J. `options` are the string-keyed
    RetrievalOptions (see julia/TadpoleRetrieval/src/common.jl).
    """
    job_id: int
    algorithm: str
    trace: np.ndarray
    dt_fs: float
    mask: np.ndarray | None = None
    delays_fs: np.ndarray | None = None
    options: dict = field(default_factory=dict)


@dataclass
class Stop:
    job_id: int


@dataclass
class Quit:
    pass


@dataclass
class Ready:
    julia_version: str
    startup_s: float


@dataclass
class Snapshot:
    job_id: int
    iteration: int
    G: float
    Gp: float
    trace: np.ndarray            # retrieved trace, μ-scaled
    E: np.ndarray                # complex field E(t)


@dataclass
class Done:
    job_id: int
    result: dict                 # E, trace, errors, errors_p, G, Gp, iterations,
                                 # stopped, t, w, algorithm, elapsed_s


@dataclass
class Error:
    job_id: int | None           # None: the worker itself failed to start
    message: str


@dataclass
class Log:
    text: str
