<p align="center"><img src="icons/Tadpole.png" alt="Tadpole" width="160"></p>

# Tadpole

Tadpole prepares SHG FROG traces and reconstructs the pulse from them. It is the companion to
**Lillypad** (acquisition) and copies its look. It reads Lillypad's `.dwc`
files, puts them on a Fourier-consistent grid, and retrieves the pulse while
showing the reconstruction live.

* **GUI:** Python 3.13, PySide6 + matplotlib (Lillypad's design system)
* **Algorithms:** Julia (`julia/TadpoleRetrieval`), running in a separate
  worker process through [juliacall](https://github.com/JuliaPy/PythonCall.jl)

| Algorithm | Status |
|---|---|
| PCGPA (principal-component generalized projections) | implemented |
| GPA (generalized projections, exact line search) | implemented |
| Ptychographic FROG (Sidorenko et al. 2016) | **skeleton**: inputs/outputs and the update equations are documented in `julia/TadpoleRetrieval/src/algorithms/ptychographic.jl` |

## Install

Needs Python 3.13 from python.org (not Anaconda) and Julia ≥ 1.10 on `PATH`
(juliaup is fine; set `TADPOLE_JULIA` to use a specific binary).

Run these from the repository root:

1. Create the virtual environment and install the Python packages:

   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   .venv\Scripts\python -m pip install -r requirements-dev.txt   # tests only
   ```

2. Install and precompile the Julia package's dependencies:

   ```powershell
   julia --project=julia/TadpoleRetrieval -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
   ```

3. Build juliacall's environment in `julia/env`. It contains PythonCall plus a
   link to `julia/TadpoleRetrieval`.

   ```powershell
   .venv\Scripts\python scripts\setup_julia.py
   ```

   This takes a few minutes the first time, because Julia precompiles
   everything. If you skip it, the app does the same thing on its first
   start, and the engine lamp stays amber until it is done.

Repeat step 2 after changing the Julia package's dependencies, and step 3
after moving the repository: `julia/env` stores absolute paths.

## Run

```powershell
.venv\Scripts\python main.py                 # or: python main.py trace.dwc
```

The Julia engine starts in the background. Watch the lamp at the bottom right: amber means starting (about 15 s), green means ready.

To try the GUI without Julia, run `$env:TADPOLE_NO_ENGINE = "1"` first.
Prepare then works fully, and Reconstruct can't start a run. If Julia fails
to start, you get the same behaviour, and the reason shows in the tooltip of
the status-bar text.

1. **Prepare.** Open a `.dwc` file (or click **Simulate**), then work through the settings:
   - Background: none, constant, or the spectrum from the edge delays.
   - Crop window.
   - Delay zero: centroid or peak of the marginal.
   - Fourier grid: N, and dt (leave it on *auto*, which balances the delay and frequency windows).

   The right-hand plots compare the measured marginals with what lands on the grid. Then click **Send to Reconstructor**.
2. **Reconstruct.** Pick an algorithm and click **Start**. The measured and retrieved traces, the difference between them, E(t), E(ω) and the convergence of **G** and **G′** update while it runs. Then save the result as `.npz` (everything) or `.csv` (the field).
3. **Analysis.** After a run: the convergence of G and G′, and measured − retrieved.
4. **Propagator.** Puts the retrieved pulse (or a saved result, **Load…**)
   through a stack of material and compares the result with a trace measured
   after that material.
   - Materials come from [refractiveindex.info](https://refractiveindex.info)
     (the `refractiveindex` package, whose database is bundled, so no
     internet access is needed). Type in the Material box to search.
   - Dispersion: the full n(ω) phase, or only its GDD (+ TOD) at the pulse
     centre. The stack's GDD and TOD are shown either way.
   - **Load measured…** takes a `.dwc` (prepared with the Prepare page's
     settings) or a prepared `.npz`, put onto the pulse's own grid.
   - SHG FROG cannot tell E(t) from its time-reversed twin E*(−t). After
     material they differ, and the page reports which one matches the
     measurement. The **direction scan** plots G against the stack's
     dispersion times s: a minimum near s = +1 confirms the retrieved
     direction (and the thickness), near −1 means the pulse is time-reversed.
     Too little material reads *inconclusive*.

### Error metrics

With I_meas normalised to peak 1 and μ the least-squares scale of the retrieved trace T:

* **G** = sqrt( Σ (I_meas − μT)² / N_pixels ). This is the standard rms FROG error (Trebino). It gets smaller on larger grids, because most pixels are empty.
* **G′** = sqrt( Σ (I_meas − μT)² / Σ I_meas² ). This is the same residual relative to the trace's own norm, so it does not depend on the grid size or on the amount of empty margin.

## Tests

```powershell
.venv\Scripts\python -m pytest                                   # Python + engine end-to-end
julia --project=julia/TadpoleRetrieval -e "using Pkg; Pkg.test()" # algorithms
```

## References

FROG and the generalized-projection algorithms (Trebino group):

1. R. Trebino and D. J. Kane, "Using phase retrieval to measure the intensity
   and phase of ultrashort pulses: frequency-resolved optical gating,"
   *J. Opt. Soc. Am. A* **10**, 1101 (1993).
   [doi:10.1364/JOSAA.10.001101](https://doi.org/10.1364/JOSAA.10.001101)
2. K. W. DeLong, B. Kohler, K. Wilson, D. N. Fittinghoff, and R. Trebino,
   "Pulse retrieval in frequency-resolved optical gating based on the method of
   generalized projections," *Opt. Lett.* **19**, 2152 (1994). **GPA**.
   [doi:10.1364/OL.19.002152](https://doi.org/10.1364/OL.19.002152)
3. R. Trebino, K. W. DeLong, D. N. Fittinghoff, J. N. Sweetser, M. A.
   Krumbügel, B. A. Richman, and D. J. Kane, "Measuring ultrashort laser pulses
   in the time-frequency domain using frequency-resolved optical gating,"
   *Rev. Sci. Instrum.* **68**, 3277–3295 (1997). Review, including the
   FROG error G.
   [doi:10.1063/1.1148286](https://doi.org/10.1063/1.1148286)
4. D. J. Kane, "Real-time measurement of ultrashort laser pulses using
   principal component generalized projections," *IEEE J. Sel. Top. Quantum
   Electron.* **4**, 278–284 (1998). **PCGPA**.
   [doi:10.1109/2944.686733](https://doi.org/10.1109/2944.686733)
5. D. J. Kane, "Recent progress toward real-time measurement of ultrashort
   laser pulses," *IEEE J. Quantum Electron.* **35**, 421–431 (1999). PCGPA in
   detail.
   [doi:10.1109/3.753647](https://doi.org/10.1109/3.753647)
6. R. Trebino, *Frequency-Resolved Optical Gating: The Measurement of
   Ultrashort Laser Pulses* (Springer/Kluwer, Boston, 2000). The FROG book.
   [doi:10.1007/978-1-4615-1181-6](https://doi.org/10.1007/978-1-4615-1181-6)

Ptychographic FROG:

7. P. Sidorenko, O. Lahav, Z. Avnat, and O. Cohen, "Ptychographic
   reconstruction algorithm for frequency-resolved optical gating:
   super-resolution and supreme robustness," *Optica* **3**, 1320 (2016).
   [doi:10.1364/OPTICA.3.001320](https://doi.org/10.1364/OPTICA.3.001320).
   Erratum: *Optica* **4**, 1388 (2017),
   [doi:10.1364/OPTICA.4.001388](https://doi.org/10.1364/OPTICA.4.001388)

## For developers

See [ARCHITECTURE.md](ARCHITECTURE.md): the folder layout, the import rules
(enforced by `tests/test_layering.py`), the Python ↔ Julia bridge, and how to
package an exe.
