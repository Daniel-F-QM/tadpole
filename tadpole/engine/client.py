"""EngineClient: the GUI's handle on the Julia worker process. No Qt.

The GUI calls `poll()` from a timer (Lillypad's display-timer pattern), so all
message handling happens on the GUI thread and no QThread is involved.

States:  "off" → "starting" → "ready" ⇄ "busy";  "error" if Julia fails to
boot. A Stop that is not honoured within STOP_GRACE_S (a Julia call that never
reaches its callback) kills the worker and starts a fresh one.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import time

from tadpole.engine import julia_worker
from tadpole.engine.protocol import (Done, Error, Quit, Ready, Snapshot, Start,
                                     Stop)

STOP_GRACE_S = 2.0


class EngineClient:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._req = None
        self._resp = None
        self._ids = itertools.count(1)
        self.state = "off"
        self.julia_version = ""
        self.job_id: int | None = None
        self._stop_sent_at: float | None = None
        self.last_error = ""

    # ── process control ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._req = self._ctx.Queue()
        self._resp = self._ctx.Queue()
        self._proc = self._ctx.Process(target=julia_worker.main,
                                       args=(self._req, self._resp),
                                       name="tadpole-julia", daemon=True)
        self._proc.start()
        self.state = "starting"
        self.job_id = None
        self._stop_sent_at = None

    def shutdown(self, timeout: float = 3.0) -> None:
        if self._proc is None:
            return
        try:
            self._req.put(Quit())
            self._proc.join(timeout)
        except (OSError, ValueError):
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(1.0)
        self._proc = None
        self.state = "off"

    def restart(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(2.0)
            self._proc = None
        self.start()

    # ── jobs ─────────────────────────────────────────────────────────────
    def submit(self, algorithm, trace, dt_fs, mask=None, delays_fs=None,
               options=None) -> int:
        if self.state != "ready":
            raise RuntimeError(f"engine is {self.state}, not ready")
        jid = next(self._ids)
        self._req.put(Start(jid, algorithm, trace, float(dt_fs), mask, delays_fs,
                            dict(options or {})))
        self.job_id = jid
        self.state = "busy"
        self._stop_sent_at = None
        return jid

    def stop(self) -> None:
        if self.state == "busy" and self.job_id is not None:
            self._req.put(Stop(self.job_id))
            self._stop_sent_at = time.monotonic()

    # ── polling ──────────────────────────────────────────────────────────
    def poll(self, max_messages: int = 200) -> list:
        """All pending messages, oldest first. Updates `state`. Snapshots of
        older jobs are dropped, and only the newest Snapshot of a burst is
        kept — the GUI only ever draws the latest."""
        out = []
        if self._resp is None:
            return out
        for _ in range(max_messages):
            try:
                msg = self._resp.get_nowait()
            except queue.Empty:
                break
            except (OSError, EOFError, ValueError):
                break
            if isinstance(msg, Ready):
                self.state = "ready"
                self.julia_version = msg.julia_version
            elif isinstance(msg, (Done, Error)):
                if msg.job_id is None:
                    self.state = "error"
                    self.last_error = msg.message
                elif msg.job_id == self.job_id:
                    self.state = "ready"
                    self.job_id = None
                    self._stop_sent_at = None
                else:
                    continue
            elif isinstance(msg, Snapshot):
                if msg.job_id != self.job_id:
                    continue
                if out and isinstance(out[-1], Snapshot):
                    out[-1] = msg
                    continue
            out.append(msg)

        if (self._stop_sent_at is not None
                and time.monotonic() - self._stop_sent_at > STOP_GRACE_S):
            jid = self.job_id
            self.restart()
            out.append(Error(jid, "The retrieval did not stop in time; the Julia "
                                  "worker was restarted."))
        if (self.state in ("starting", "ready", "busy") and self._proc is not None
                and not self._proc.is_alive()):
            self.state = "error"
            self.last_error = (f"Julia worker exited (code {self._proc.exitcode}); "
                               f"see {julia_worker.worker_log_path()}")
            out.append(Error(None, self.last_error))
        return out
