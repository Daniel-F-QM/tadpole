"""
    TadpoleRetrieval

FROG pulse-retrieval algorithms for Tadpole.

Conventions used everywhere in this package
-------------------------------------------
* Times in fs, angular frequencies in rad/fs.
* A trace is a matrix `I[ω, τ]`: ROWS are frequency, COLUMNS are delay — the
  same orientation as a .dwc file (wavelength rows × delay columns).
* Frequencies are OFFSETS from the trace's centre frequency; the absolute
  carrier is bookkeeping on the Python side.
* The time → frequency transform uses the kernel e^{-iωt} (Eq. 1 of Sidorenko
  et al. 2016), implemented as `fftshift(fft(ifftshift(x)))` along dim 1.
* SHG signal: E_sig(t, τ) = E(t) E(t − τ).

The one entry point Python calls is [`run_retrieval`](@ref). It takes and
returns only plain arrays, numbers and Dicts, so the bridge can be swapped
(juliacall today, a compiled stdin/stdout app later) without touching the
algorithms.
"""
module TadpoleRetrieval

using FFTW
using LinearAlgebra
using Random
using PrecompileTools

export FourierGrid, SHG, Nonlinearity
export simulate_pulse, frog_trace, run_retrieval, RetrievalResult
export retrieve_pcgpa, retrieve_gpa, retrieve_ptychographic
export frog_error, pulse_distance, center_pulse

include("grid.jl")
include("nonlinearity.jl")
include("common.jl")
include("simulate.jl")
include("algorithms/pcgpa.jl")
include("algorithms/gpa.jl")
include("algorithms/ptychographic.jl")
include("api.jl")
include("precompile.jl")

end # module
