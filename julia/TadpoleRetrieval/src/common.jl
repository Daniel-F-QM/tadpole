# Shared machinery: options, results, the FROG error, the intensity
# projection, initial guesses, ambiguity handling and progress reporting.

"""
    RetrievalOptions

Every algorithm reads the fields it understands and ignores the rest.

* `max_iter`        hard iteration cap
* `tol`             stop once the FROG error G drops below this
* `snapshot_every`  report progress every this many iterations…
* `min_snapshot_s`  …but not more often than this (seconds), so the GUI is not
                    flooded on small grids
* `init`            :autocorrelation | :gaussian | :random
* `init_fwhm_fs`    FWHM of the :gaussian guess (≤ 0 → derived from the trace)
* `seed`            RNG seed, for reproducible runs
* `alpha_min`, `alpha_max`, `gamma`   ptychographic step size range and
                    soft-threshold level (Sidorenko et al. 2016)
"""
Base.@kwdef struct RetrievalOptions
    max_iter::Int = 500
    tol::Float64 = 1e-5
    snapshot_every::Int = 1
    min_snapshot_s::Float64 = 0.1
    init::Symbol = :autocorrelation
    init_fwhm_fs::Float64 = 0.0
    seed::Int = 0
    alpha_min::Float64 = 0.1
    alpha_max::Float64 = 0.5
    gamma::Float64 = 0.0
end

"""
    RetrievalResult

* `E`          retrieved field E(t) on the grid, centred in time, peak |E| = 1
* `trace`      retrieved trace, scaled (μ) to best match the measured one
* `errors`     FROG error G after every iteration
* `errors_p`   normalised error G′ after every iteration
* `G`, `Gp`    G and G′ of the best (lowest-G) iterate — `E` is that field
* `iterations` iterations actually run
* `stopped`    "tolerance" | "max_iter" | "user"
"""
struct RetrievalResult
    E::Vector{ComplexF64}
    trace::Matrix{Float64}
    errors::Vector{Float64}
    errors_p::Vector{Float64}
    G::Float64
    Gp::Float64
    iterations::Int
    stopped::String
end

"""Measured trace normalised to peak 1, as Float64, negatives clipped."""
function normalize_trace(I::AbstractMatrix)
    T = Matrix{Float64}(I)
    T .= max.(T, 0.0)
    m = maximum(T)
    m > 0 || throw(ArgumentError("trace is empty (all ≤ 0)"))
    T ./ m
end

"""Least-squares scale μ minimising Σ (I_meas − μ·T)² over the measured pixels."""
function best_scale(Imeas::AbstractMatrix, T::AbstractMatrix, mask)
    num = 0.0; den = 0.0
    @inbounds for j in eachindex(T)
        (mask === nothing || mask[j]) || continue
        num += Imeas[j] * T[j]
        den += T[j]^2
    end
    den > 0 ? num / den : 1.0
end

"""
    frog_error(Imeas, T; mask=nothing) -> (G, Gp, μ)

Both errors over the measured pixels, with I_meas normalised to peak 1 and μ
the least-squares scale of the retrieved trace T:

* G  = sqrt( Σ (I_meas − μT)² / n )          the standard rms FROG error
       (Trebino), which shrinks as the grid grows because most pixels are
       empty;
* G′ = sqrt( Σ (I_meas − μT)² / Σ I_meas² )  the normalised error, the same
       residual relative to the trace's own norm — independent of grid size
       and of how much empty margin surrounds the trace.
"""
function frog_error(Imeas::AbstractMatrix, T::AbstractMatrix; mask=nothing)
    μ = best_scale(Imeas, T, mask)
    s = 0.0; n = 0; s2 = 0.0
    @inbounds for j in eachindex(T)
        (mask === nothing || mask[j]) || continue
        s += (Imeas[j] - μ * T[j])^2
        s2 += Imeas[j]^2
        n += 1
    end
    sqrt(s / max(n, 1)), sqrt(s / max(s2, eps())), μ
end

"""
    intensity_projection!(Esig, Imeas, μ, mask)

Replace |E_sig(ω, τ)| by sqrt(I_meas/μ), keeping the phase. Pixels outside the
mask (unmeasured) keep the computed value. In place.
"""
function intensity_projection!(Esig::AbstractMatrix{ComplexF64}, Imeas::AbstractMatrix,
                               μ::Real, mask)
    invμ = 1 / max(μ, eps())
    @inbounds for j in eachindex(Esig)
        (mask === nothing || mask[j]) || continue
        a = abs(Esig[j])
        amp = sqrt(Imeas[j] * invμ)
        Esig[j] = a > 0 ? Esig[j] * (amp / a) : complex(amp)
    end
    Esig
end

# ── Initial guesses ──────────────────────────────────────────────────────────

"""FWHM of a sampled curve, by linear interpolation of the half-max crossings."""
function fwhm(x::AbstractVector, y::AbstractVector)
    m, ip = findmax(y)
    m > 0 || return 0.0
    h = m / 2
    l = ip
    while l > 1 && y[l] > h; l -= 1; end
    r = ip
    while r < length(y) && y[r] > h; r += 1; end
    xl = y[l] <= h && l < ip ? x[l] + (h - y[l]) / (y[l+1] - y[l]) * (x[l+1] - x[l]) : x[l]
    xr = y[r] <= h && r > ip ? x[r-1] + (y[r-1] - h) / (y[r-1] - y[r]) * (x[r] - x[r-1]) : x[r]
    xr - xl
end

"""
    initial_guess(Imeas, grid, opts, rng)

* :autocorrelation  sqrt of the delay marginal (∝ intensity autocorrelation
                    for SHG), with a small random phase — the guess
                    Sidorenko et al. use
* :gaussian         a Gaussian whose FWHM is `init_fwhm_fs`, or the
                    autocorrelation FWHM/√2 when that is ≤ 0
* :random           random amplitude and phase (Kane's PCGPA start)
"""
function initial_guess(Imeas::AbstractMatrix, grid::FourierGrid, opts::RetrievalOptions,
                       rng::AbstractRNG)
    N = grid.N
    marg = vec(sum(Imeas; dims=1))
    phase_noise = exp.(im .* 0.1 .* randn(rng, N))
    if opts.init == :random
        E = rand(rng, N) .* exp.(2π * im .* rand(rng, N))
    elseif opts.init == :gaussian
        w = opts.init_fwhm_fs > 0 ? opts.init_fwhm_fs : fwhm(grid.t, marg) / sqrt(2)
        w = w > 0 ? w : 10 * grid.dt
        E = exp.(-2log(2) .* (grid.t ./ w) .^ 2) .* phase_noise
    elseif opts.init == :autocorrelation
        E = sqrt.(max.(marg, 0.0)) .* phase_noise
    else
        throw(ArgumentError("unknown init $(opts.init)"))
    end
    ComplexF64.(E ./ maximum(abs.(E)))
end

# ── Ambiguities ──────────────────────────────────────────────────────────────

"""
    center_pulse(E, grid)

Shift E so its intensity centroid sits at t = 0 (sub-sample, via the Fourier
shift theorem), remove the global phase at the peak and normalise peak |E| = 1.
"""
function center_pulse(E::AbstractVector, grid::FourierGrid)
    I = abs2.(E)
    tc = sum(grid.t .* I) / sum(I)
    Ec = shift_field(E, grid, -tc)
    _, ip = findmax(abs.(Ec))
    Ec .*= exp(-im * angle(Ec[ip]))
    Ec ./ maximum(abs.(Ec))
end

"""Angle between two fields, insensitive to global phase (Eq. 9, Sidorenko)."""
function field_angle(a::AbstractVector, b::AbstractVector)
    c = abs(dot(a, b)) / sqrt(real(dot(a, a)) * real(dot(b, b)))
    acos(clamp(c, 0.0, 1.0))
end

"""
    pulse_distance(E, Eref, grid)

The angle δ between `E` and `Eref` after removing the SHG ambiguities: time
shift (both centred), global phase, and time reversal E(t) → E*(−t).
"""
function pulse_distance(E::AbstractVector, Eref::AbstractVector, grid::FourierGrid)
    a = center_pulse(E, grid)
    b = center_pulse(Eref, grid)
    # Time reversal on this grid: t_k ↔ −t_k maps index k to N+2−k (index 1,
    # t = −N/2·dt, has no partner and is taken to itself).
    N = grid.N
    rev = [conj(a[k == 1 ? 1 : N + 2 - k]) for k in 1:N]
    rev = center_pulse(rev, grid)
    min(field_angle(a, b), field_angle(rev, b))
end

# ── Progress ─────────────────────────────────────────────────────────────────

"""
Throttled progress reporting. `callback(iter, G, Gp, trace, E)` returns `true` to
request a stop. `nothing` disables reporting.
"""
mutable struct Reporter
    callback::Any
    every::Int
    min_s::Float64
    last::Float64
end

Reporter(cb, opts::RetrievalOptions) =
    Reporter(cb, max(opts.snapshot_every, 1), opts.min_snapshot_s, 0.0)

"""Report if due (or `force`); returns true when the caller asked to stop."""
function report!(r::Reporter, iter::Int, G::Float64, Gp::Float64, make_trace, E; force=false)
    r.callback === nothing && return false
    now = time()
    due = force || (iter % r.every == 0 && now - r.last >= r.min_s)
    due || return false
    r.last = now
    stop = r.callback(iter, G, Gp, make_trace(), E)
    stop === true
end

"""Bookkeeping common to the iterative loops: best field, error history."""
mutable struct Tracker
    bestE::Vector{ComplexF64}
    bestG::Float64
    bestGp::Float64
    errors::Vector{Float64}
    errors_p::Vector{Float64}
end

Tracker(E) = Tracker(copy(E), Inf, Inf, Float64[], Float64[])

function record!(tr::Tracker, E, G, Gp)
    push!(tr.errors, G)
    push!(tr.errors_p, Gp)
    if G < tr.bestG
        tr.bestG = G
        tr.bestGp = Gp
        tr.bestE .= E
    end
end

"""Package the best field of a finished run as a RetrievalResult."""
function finish(nl::Nonlinearity, tr::Tracker, Imeas, grid, mask, iters, stopped)
    E = center_pulse(tr.bestE, grid)
    T = frog_trace(nl, E)
    _, _, μ = frog_error(Imeas, T; mask=mask)
    RetrievalResult(E, μ .* T, tr.errors, tr.errors_p, tr.bestG, tr.bestGp, iters, stopped)
end
