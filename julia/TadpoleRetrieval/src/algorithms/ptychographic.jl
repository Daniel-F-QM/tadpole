# Ptychographic FROG retrieval — SKELETON, to be implemented.
#
# Sidorenko, Lahav, Avnat, Cohen, "Ptychographic reconstruction algorithm for
# frequency-resolved optical gating: super-resolution and supreme robustness",
# Optica 3, 1320 (2016). Erratum: Optica 4, 1388 (2017).
#
# Unlike GPA/PCGPA this method updates the field from ONE measured spectrum
# at a time, so the delay axis is decoupled from the time grid:
#   * delays need not be uniform, nor equal to the time step, nor N in number
#   * unmeasured frequencies (mask Ω false) are allowed — "super-resolution"
# That is why it takes the measured `delays` explicitly instead of a square
# trace.

"""
    retrieve_ptychographic(Imeas, delays, dt; mask=nothing,
                           opts=RetrievalOptions(), nl=SHG(),
                           callback=nothing) -> RetrievalResult

**Inputs**

* `Imeas::AbstractMatrix` — measured trace `I[ω, j]`, size N × J: N frequency
  rows on `FourierGrid(N, dt).w`, J delay columns. Need not be square.
* `delays::AbstractVector` — the J measured delays τ_j in fs, any order and
  spacing (non-integer multiples of `dt` allowed → use `shift_field`).
* `dt::Real` — time step of the reconstruction grid (fs); sets N·dt window.
* `mask` — optional `BitMatrix`/`Matrix{Bool}` N × J; `true` = reliably
  measured pixel (the set Ω). Pixels outside Ω are only soft-thresholded.
* `opts` — reads `max_iter`, `tol`, `alpha_min`/`alpha_max` (step size drawn
  uniformly per iteration, paper: [0.1, 0.5]), `gamma` (soft threshold, paper:
  1e-3…1e-6, 0 = off), `init`, `seed`, `snapshot_every`, `min_snapshot_s`.
* `callback(iter, G, Gp, trace, E)` — progress, return `true` to stop. `trace`
  must be N × J (same shape as `Imeas`).

**Output** — a [`RetrievalResult`](@ref):

* `E` — N-vector E(t) on the grid, `center_pulse`d, peak |E| = 1
* `trace` — N × J retrieved trace at the measured delays, μ-scaled
* `errors`, `errors_p` — G and G′ per iteration (over Ω, see `frog_error`)
* `G`, `Gp`, `iterations`, `stopped` ("tolerance" | "max_iter" | "user")

**Algorithm** (one iteration; j runs over a fresh random permutation s(j)):

    α ~ U[alpha_min, alpha_max]                            (once per iteration)
    for j in randperm(J)
        Eτ    = E(t − τ_j)                                 (shift_field)
        ψ     = E · Eτ                                     (Eq. 3)
        Ψ     = t2w(ψ)
        Φ     = sqrt(I_j) · Ψ/|Ψ|        on Ω              (Eq. 4)
        Φ     = f_γ(Re Ψ) + i f_γ(Im Ψ)  on Ωᶜ             (Eq. 5, 6; soft threshold)
        ψ'    = w2t(Φ)                                     (Eq. 7)
        E   ← E + α · conj(Eτ) / max|Eτ|² · (ψ' − ψ)        (Eq. 8)
    end
    G, G′ over all J columns (frog_error), record!, report!, tolerance check

Initial guess: `initial_guess(...; init=:autocorrelation)` — the paper's
choice. The measured trace needs normalising so √I_j and |Ψ| share a scale
(e.g. via `best_scale` once per iteration).
"""
function retrieve_ptychographic(Imeas::AbstractMatrix, delays::AbstractVector, dt::Real;
                                mask=nothing, opts::RetrievalOptions=RetrievalOptions(),
                                nl::Nonlinearity=SHG(), callback=nothing)
    size(Imeas, 2) == length(delays) ||
        throw(DimensionMismatch("trace has $(size(Imeas, 2)) delay columns, " *
                                "but $(length(delays)) delays were given"))
    error("ptychographic retrieval is not implemented yet")
end

"""Soft threshold f_γ(x) of Eq. 6: 0 inside ±γ, shrunk towards 0 by γ outside."""
soft_threshold(x::Real, γ::Real) = abs(x) < γ ? zero(x) : x - γ * sign(x)
