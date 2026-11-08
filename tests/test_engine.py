"""End to end through the Julia worker process (skipped if Julia is missing)."""

import time

import pytest

from tadpole.analysis import pulse_views
from tadpole.engine.protocol import Done, Error, Snapshot
from tadpole.prep.model import PrepSettings
from tadpole.prep.pipeline import prepare
from tadpole.prep.simulate import simulate_raw

pytestmark = pytest.mark.timeout(400)


def run(engine, alg, trace, dt, timeout=120, **kw):
    engine.submit(alg, trace, dt, **kw)
    snaps, final = [], None
    t0 = time.monotonic()
    while final is None and time.monotonic() - t0 < timeout:
        for m in engine.poll():
            if isinstance(m, Snapshot):
                snaps.append(m)
            elif isinstance(m, (Done, Error)):
                final = m
        time.sleep(0.02)
    return snaps, final


@pytest.mark.parametrize("alg", ["pcgpa", "gpa"])
def test_retrieves_simulated_measurement(engine, alg):
    # 30 fs transform-limited pulse with 600 fs² GDD → 63.0 fs FWHM
    raw = simulate_raw(noise=0.01)
    p, _ = prepare(raw, PrepSettings(grid_n=128))
    snaps, done = run(engine, alg, p.trace, p.dt_fs, mask=p.mask,
                      options={"max_iter": 400})
    assert isinstance(done, Done), done
    r = done.result
    assert r["G"] < 5e-3 and r["Gp"] < 5e-2
    assert len(r["errors"]) == len(r["errors_p"]) == r["iterations"]
    assert snaps and snaps[-1].trace.shape == (128, 128)
    v = pulse_views(r["E"], r["t"], r["w"], p.w0_rad_fs)
    assert v["fwhm_t"] == pytest.approx(63.0, rel=0.05)
    assert v["lambda0_nm"] == pytest.approx(800.0, abs=2)


def test_stop(engine):
    p, _ = prepare(simulate_raw(), PrepSettings(grid_n=64))
    engine.submit("gpa", p.trace, p.dt_fs, options={"max_iter": 10**7, "tol": 0.0})
    time.sleep(1.0)
    engine.stop()
    t0 = time.monotonic()
    done = None
    while done is None and time.monotonic() - t0 < 10:
        done = next((m for m in engine.poll() if isinstance(m, (Done, Error))), None)
        time.sleep(0.02)
    assert isinstance(done, Done) and done.result["stopped"] == "user"
    assert engine.state == "ready"


def test_ptychographic_reports_not_implemented(engine):
    p, _ = prepare(simulate_raw(), PrepSettings(grid_n=64))
    _, final = run(engine, "ptychographic", p.trace_meas, p.dt_fs,
                   mask=p.mask_meas, delays_fs=p.delays_meas_fs)
    assert isinstance(final, Error) and "not implemented" in final.message
    assert engine.state == "ready"


def test_python_trace_model_matches_julia(engine):
    """The Propagator's numpy SHG model must be Julia's frog_trace exactly,
    or its comparisons would carry a convention error."""
    from tadpole.propagation.physics import shg_trace, trace_errors
    p, _ = prepare(simulate_raw(noise=0.0), PrepSettings(grid_n=64))
    _, done = run(engine, "pcgpa", p.trace, p.dt_fs, mask=p.mask,
                  options={"max_iter": 50})
    assert isinstance(done, Done), done
    r = done.result
    assert trace_errors(r["trace"], shg_trace(r["E"]))[1] < 1e-6
