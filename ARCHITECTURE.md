# Tadpole architecture

## Layout

```
main.py                     launcher (python main.py [trace.dwc])
Tadpole.spec                PyInstaller recipe (one-folder exe + Julia bundle)
requirements*.txt           pinned Python deps (runtime / tests)
scripts/
  setup_julia.py            builds julia/env (juliacall's environment)
  build_julia_bundle.py     self-contained Julia runtime+depot+env for the exe
  build_sysimage.jl         PackageCompiler sysimage for that bundle
icons/                      PNGs copied from Lillypad
julia/TadpoleRetrieval/     the Julia package — all retrieval maths
  src/grid.jl               FourierGrid, t2w/w2t, shift_field
  src/nonlinearity.jl       Nonlinearity, SHG; signal matrix, frog_trace
  src/common.jl             options, result, G/G′, projection, guesses, ambiguities
  src/simulate.jl           test pulses
  src/algorithms/           pcgpa.jl, gpa.jl, ptychographic.jl (skeleton)
  src/api.jl                run_retrieval — the ONLY entry point Python calls
  src/precompile.jl         PrecompileTools workload
  test/runtests.jl
julia/env/                  (generated, git-ignored) juliacall's environment
tadpole/
  paths.py                  the only __file__ anchor; where Julia lives
  settings.py               settings.json (Lillypad's mechanism)
  analysis.py               FWHM, spectra, blanked phases (display physics)
  propagation/              materials.py (refractiveindex.info, the ONLY importer
                            of `refractiveindex`), physics.py (stack phase,
                            propagation, numpy SHG model, direction scan)
  juliapkg.json             tells juliapkg to dev julia/TadpoleRetrieval
  design/                   Lillypad's design system: tokens, qss, theme, mpl
  io/                       dwc.py (read/write .dwc), prepared.py (.npz/.csv)
  prep/                     background, crop, grid (λ→ω, resampling), pipeline,
                            model (PrepSettings, PreparedTrace), simulate
  engine/                   protocol (messages), julia_worker (the process),
                            client (EngineClient, polled by the GUI)
  gui/                      widgets, state, canvas/, controllers/, dialogs/,
                            window/ (build_ui + TadpoleWindow)
    canvas/                 base (styling), interaction (Lillypad's mouse
                            gestures), prep_canvas, recon_canvas, analysis_canvas
tests/                      pytest (offscreen GUI, layering, engine end-to-end)
```

## Layering rules (tests/test_layering.py)

* `design/` never imports `gui/`.
* `prep/`, `io/` and `engine/` import no Qt and no matplotlib. They are
  testable headlessly and reusable from scripts.
* **Only `engine/julia_worker.py` imports juliacall.** Julia never runs in the
  GUI process.

## GUI

Four pages: Prepare, Reconstruct, Analysis and Propagator. Analysis unlocks
when a retrieval finishes; the Propagator takes that pulse or a saved one. The control columns have a fixed height and never scroll.
Explanations are tooltips, and rarely used options sit in each page's
Advanced… window (`dialogs/advanced.py`). Those widgets keep their usual names
on the window, so persistence doesn't care where they are shown.

Every plot works like Lillypad's (`canvas/interaction.py`):

* drag to zoom;
* right-click to return to the automatic view;
* click an end tick label to type a bound;
* on a colour bar, drag or scroll to set the range.

Canvases register their panels and colour bars in `redraw()`. User zooms are
stored per key, so they survive a figure rebuild. Measured and Retrieved share
their axes and one colour range.

## Propagator

* Sign convention: with the e^{-iωt} kernel, material multiplies Ẽ(ω) by
  e^{−iφ(ω)}, φ = n(ω)·ω·L/c, so positive GDD delays blue behind red. Pulse
  GDDs are reported in the same sense (positive = like glass). The constant
  and the group delay at ω₀ are always removed.
* Outside a page's tabulated range the index is NaN, never extrapolated.
  Those frequencies take the cubic Taylor expansion at ω₀, and the panel
  warns when the pulse spectrum has energy there.
* The comparison trace is put onto the pulse's own grid: `prepare(raw, s,
  grid=(n, dt, w0))` for a .dwc and `regrid` for a prepared .npz.
* `physics.shg_trace` is a numpy copy of Julia's `frog_trace`. An engine test
  checks they agree, so the Propagator's comparisons and the retrieval use
  the same model.
* The direction scan runs on a worker thread (numpy FFTs release the GIL) and
  is polled. A scan whose inputs have changed is dropped.

## Data flow

```
.dwc ─read_dwc→ RawTrace ─prepare(PrepSettings)→ PreparedTrace ─EngineClient.submit→
   worker process: TadpoleRetrieval.run_retrieval(alg, trace, dt; mask, delays, options, callback)
      ├─ callback every k iterations (throttled ≥ 0.1 s) → Snapshot → queue → GUI
      └─ result Dict → Done → queue → GUI
```

`PreparedTrace` carries two views of the data:

* `trace`/`mask`: the N×N grid for GPA/PCGPA (delay step = time step,
  Δt·Δν = 1/N).
* `trace_meas`/`mask_meas`/`delays_meas_fs`: resampled in frequency only, at
  the measured delays. This is what the ptychographic algorithm will use.

### Preparation details worth knowing

* λ→ω conversion multiplies by λ²/(2πc), so ∫I dω = ∫I dλ.
* When the target grid is coarser than the data, values are bin averages
  computed from the cumulative integral. When it is finer, they are linearly
  interpolated.
* **Negative pixels are kept until after binning.** If you clip noise to zero
  first, it becomes a positive floor under the whole trace. On a test trace
  with 1% noise that made the retrieved pulse about 10% too short.
* Auto dt is the geometric mean of the two bounds: N·dt must cover the delay
  signal, and 2π/dt must cover the spectrum. That leaves the same headroom on
  both axes.
* The report includes the **delay asymmetry** ‖T(τ)−T(−τ)‖/‖T‖. An SHG trace
  is symmetric, so a large value means an SHG retrieval cannot fit part of the
  data.

## Julia conventions

* Trace `I[ω, τ]`: rows are frequency, columns are delay (the .dwc
  orientation).
* t→ω uses `fftshift(fft(ifftshift(x)))` (kernel e^{-iωt}). Python's
  `tadpole.analysis.t2w` is identical.
* SHG signal: `S[i,k] = E[i]·E[i−m_k]` (circular), with `τ_k = m_k·dt`.
* Errors: G = sqrt(Σ(I−μT)²/n) and G′ = sqrt(Σ(I−μT)²/ΣI²), both with the
  least-squares μ.
* Callback signature: `callback(iter, G, Gp, trace, E) -> Bool`. Returning
  `true` stops the run.

To add a nonlinearity (THG, PG…), subtype `Nonlinearity`, give it a `gate`
method, and teach the PCGPA power step about it. GPA and the forward model
already go through `signal_matrix`.

To implement the ptychographic algorithm, fill in the body of
`retrieve_ptychographic` (its docstring gives the inputs, the outputs and
Eqs. 3–8). The GUI, the bridge and the tests already route to it. Then flip
`test_ptychographic_reports_not_implemented` into a convergence test.

## The Python ↔ Julia bridge

* `EngineClient.start()` starts `julia_worker.main` with multiprocessing
  "spawn". The worker:
  1. sets the variables from `paths.julia_environment()`;
  2. imports juliacall and runs `using TadpoleRetrieval`;
  3. warms up the methods for numpy-backed arrays (otherwise the first real
     call spends about 8 s compiling);
  4. sends `Ready`.

  Startup takes about 15 s and runs in the background. The lamp at the bottom
  right of the window shows it.
* The GUI polls with a 60 ms `QTimer`. There is no QThread. Of a burst of
  snapshots, only the newest is drawn.
* Stop is cooperative: the callback drains the request queue for a `Stop`. If
  the run is not done within 2 s (stuck inside a Julia call), the worker is
  terminated and respawned.
* Python dicts arrive in Julia as `PyDict{Any,Any}`. The worker converts them
  to `Dict{String,Any}` before calling `run_retrieval`.

## Packaging (exe)

PyInstaller cannot bundle Julia by itself, so the exe ships a self-contained
**Julia bundle** at `_internal/julia_bundle`:

| part | size | what |
|---|---|---|
| `runtime/` | ~1.0 GB | a copy of the Julia install (minus docs/headers) |
| `depot/` | ~0.35 GB | packages + precompile caches; TadpoleRetrieval lives in `depot/dev` |
| `env/` | tiny | Project/Manifest with relative paths |
| `sysimage/tadpole.dll` | see below | optional PackageCompiler image |

When frozen (or when `TADPOLE_JULIA_BUNDLE` points at a bundle),
`paths.julia_environment()` sets `PYTHON_JULIACALL_EXE` and
`PYTHON_JULIACALL_PROJECT`. juliacall then skips juliapkg completely: nothing
is resolved or downloaded on the user's machine, and no Julia install is
needed there. `JULIA_DEPOT_PATH` lists a writable user depot
(`%LOCALAPPDATA%\Tadpole\julia_depot`) before the bundled one.

Build:

```powershell
.venv\Scripts\python scripts\build_julia_bundle.py
julia --startup-file=no scripts\build_sysimage.jl
.venv\Scripts\pyinstaller Tadpole.spec --noconfirm
```

Check a build without clicking through it:
`dist\Tadpole\Tadpole.exe --selftest report.json`. This boots the engine,
retrieves a simulated 63 fs pulse with PCGPA, and writes a JSON report. The
exit code is 0 on success.

### Findings from the packaging spike (Nov 2026, Julia 1.11.9)

| bundle setup | first start on a "new machine" (moved bundle, empty user depot, no Julia on PATH) |
|---|---|
| precompile caches only | 216 s: FFTW, MKL_jll, TadpoleRetrieval… are rejected after the move and recompiled into the user depot |
| + sysimage | 30 s, **but** it downloaded MKL (lazy artifact) on first start, which fails offline |
| + sysimage + lazy artifacts bundled | **18 s, nothing compiled or downloaded** ✔ |

* The bundle is about 2.1 GB: runtime 1.0, depot 0.8 (MKL alone is 0.4,
  pulled in by FFTW even though FFTW's default provider doesn't use it), and
  sysimage 0.3.
* Possible trims, not done yet: drop MKL by pinning FFTW to its FFTW_jll
  provider and leaving MKL_jll out of the sysimage; strip unused parts of the
  runtime (`lib/julia/*.a`, `share/julia/test`).
* PackageCompiler's MinGW (~0.7 GB) must not land in the bundle depot. That is
  why `build_sysimage.jl` *appends* the bundle depot to `DEPOT_PATH`.
* The VS Code Julia extension indexes new environments under `build/` and can
  hold file locks there for a while. Moving or deleting `build/julia_bundle`
  right after a build can fail with "permission denied".
