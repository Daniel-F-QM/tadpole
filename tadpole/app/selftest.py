"""`Tadpole.exe --selftest [report.json]`: check a build without the GUI.

Boots the Julia engine exactly as the window would, prepares a simulated trace,
retrieves it with PCGPA and writes a JSON report (the frozen exe has no
console, so stdout is not an option). Exit code 0 = everything worked.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from tadpole.paths import app_dir


def run(report_path: str | None = None) -> int:
    path = Path(report_path) if report_path else app_dir() / "selftest.json"
    report: dict = {"ok": False, "frozen": bool(getattr(sys, "frozen", False))}
    t0 = time.monotonic()
    client = None
    try:
        from tadpole.analysis import pulse_views
        from tadpole.engine.client import EngineClient
        from tadpole.engine.protocol import Done, Error
        from tadpole.paths import julia_environment
        from tadpole.prep.model import PrepSettings
        from tadpole.prep.pipeline import prepare
        from tadpole.prep.simulate import simulate_raw

        report["julia_env"] = julia_environment()
        client = EngineClient()
        client.start()
        while client.state == "starting" and time.monotonic() - t0 < 600:
            client.poll()
            time.sleep(0.05)
        report["engine_state"] = client.state
        report["engine_ready_s"] = round(time.monotonic() - t0, 1)
        if client.state != "ready":
            report["error"] = client.last_error
        else:
            p, _ = prepare(simulate_raw(noise=0.01), PrepSettings(grid_n=128))
            client.submit("pcgpa", p.trace, p.dt_fs, mask=p.mask,
                          options={"max_iter": 300})
            final = None
            while final is None and time.monotonic() - t0 < 900:
                final = next((m for m in client.poll()
                              if isinstance(m, (Done, Error))), None)
                time.sleep(0.05)
            if isinstance(final, Done):
                r = final.result
                v = pulse_views(r["E"], r["t"], r["w"], p.w0_rad_fs)
                report.update(G=r["G"], Gp=r["Gp"], iterations=r["iterations"],
                              fwhm_fs=v["fwhm_t"], expected_fwhm_fs=63.0)
                report["ok"] = abs(v["fwhm_t"] - 63.0) < 3.0
            else:
                report["error"] = getattr(final, "message", "timed out")
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if client is not None:
            client.shutdown()
    report["total_s"] = round(time.monotonic() - t0, 1)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if report["ok"] else 1
