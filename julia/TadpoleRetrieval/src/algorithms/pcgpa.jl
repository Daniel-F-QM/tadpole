# Principal-component generalized projections (Kane, IEEE JQE 35, 421 (1999)).
#
# One iteration:
#   1. Build the signal matrix S(t, τ) = E(t) G(t − τ) from the current field.
#      It is a re-indexing of the outer product O = E Gᵀ: every (t_i, τ_k)
#      pixel is exactly one O[i, j] with j = i − m_k (mod N).
#   2. FT along t, impose the measured magnitude, inverse FT  →  S'.
#   3. Undo the re-indexing  →  O'.
#   4. The new field is the principal component of O'. For SHG O = E Eᵀ is
#      complex SYMMETRIC (not Hermitian), and O·E* = E·‖E‖², so one power
#      step on the symmetrised O' is E ← O'_sym · E*.
#
# Circular indexing is inherent to PCGPA; the prepared trace must have enough
# empty margin in delay and frequency that wrap-around carries no signal.

"""
    outer_index(N) -> Matrix{Int}

`idx[i, k]` is the linear index into the N×N outer product O[i, j] that the
signal-matrix pixel S[i, k] = E_i · G_{i − m_k} reads. It is a permutation of
1:N², so the same table maps S → O and back.
"""
function outer_index(N::Integer)
    m = delay_offsets(N)
    idx = Matrix{Int}(undef, N, N)
    @inbounds for k in 1:N, i in 1:N
        j = mod1(i - m[k], N)
        idx[i, k] = i + (j - 1) * N          # column-major linear index of O[i, j]
    end
    idx
end

"""
    retrieve_pcgpa(Imeas, dt; mask=nothing, opts=RetrievalOptions(),
                   nl=SHG(), callback=nothing) -> RetrievalResult

PCGPA on an N×N trace `Imeas[ω, τ]` sampled on `FourierGrid(N, dt)` (the delay
step equals the time step). `mask[ω, τ] = false` marks unmeasured pixels,
which keep their computed value. `callback(iter, G, Gp, trace, E)` is called
(throttled) during the run and may return `true` to stop.
"""
function retrieve_pcgpa(Imeas_in::AbstractMatrix, dt::Real; mask=nothing,
                        opts::RetrievalOptions=RetrievalOptions(),
                        nl::Nonlinearity=SHG(), callback=nothing)
    nl isa SHG || throw(ArgumentError("PCGPA is only implemented for SHG so far"))
    N = size(Imeas_in, 1)
    size(Imeas_in, 2) == N || throw(ArgumentError("PCGPA needs a square N×N trace"))
    grid = FourierGrid(N, dt)
    Imeas = normalize_trace(Imeas_in)
    rng = MersenneTwister(opts.seed)
    E = initial_guess(Imeas, grid, opts, rng)

    idx = outer_index(N)
    O = Matrix{ComplexF64}(undef, N, N)
    S = Matrix{ComplexF64}(undef, N, N)
    tr = Tracker(E)
    rep = Reporter(callback, opts)
    stopped = "max_iter"
    it = 0
    for outer it in 1:opts.max_iter
        # 1. signal matrix from the outer product
        mul!(O, E, transpose(E))
        @inbounds for q in eachindex(S); S[q] = O[idx[q]]; end
        # 2. intensity constraint
        Esig = t2w(S)
        G, Gp, μ = frog_error(Imeas, abs2.(Esig); mask=mask)
        record!(tr, E, G, Gp)
        if report!(rep, it, G, Gp, () -> μ .* abs2.(Esig), E)
            stopped = "user"; break
        end
        if G < opts.tol
            stopped = "tolerance"; break
        end
        intensity_projection!(Esig, Imeas, μ, mask)
        S .= w2t(Esig)
        # 3. back to the outer-product form
        @inbounds for q in eachindex(S); O[idx[q]] = S[q]; end
        # 4. principal component (one power step, symmetric for SHG)
        Osym = (O .+ transpose(O)) ./ 2
        Enew = Osym * conj.(E)
        nrm = maximum(abs.(Enew))
        nrm > 0 || break
        E = Enew ./ nrm
        # keep the pulse from drifting around the circular grid
        E = recentre_integer(E)
    end
    res = finish(nl, tr, Imeas, grid, mask, it, stopped)
    report!(rep, it, res.G, res.Gp, () -> res.trace, res.E; force=true)
    res
end

"""Circularly shift E by a whole number of samples so its centroid is near t = 0."""
function recentre_integer(E::AbstractVector)
    N = length(E)
    I = abs2.(E)
    s = sum(I)
    s > 0 || return E
    # centroid of a circular distribution, so a pulse straddling the edge
    # is not "centred" at the middle of the empty region
    θ = 2π .* (0:N-1) ./ N
    c = angle(sum(I .* exp.(im .* θ)))
    pos = mod(c / 2π * N, N)                 # 0-based index of the centroid
    shift = round(Int, N ÷ 2 - pos)
    circshift(E, shift)
end
