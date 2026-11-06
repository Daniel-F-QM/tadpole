# Generalized projections (DeLong, Fittinghoff, Trebino, Opt. Lett. 19, 2152
# (1994); Trebino's book, ch. 8).
#
# One iteration:
#   1. Signal field from the current pulse, impose the measured magnitude,
#      back to time  →  S'(t, τ)  (the "data constraint" projection).
#   2. Project onto the "mathematical form" constraint S = E(t)E(t − τ) by
#      minimising  Z(E) = Σ_{t,τ} |S'(t, τ) − E(t) E(t − τ)|²  along its
#      steepest-descent direction. Z(E − α g) is a real QUARTIC in α, so the
#      line search is exact: the minimum is a root of its cubic derivative.

"""Wirtinger gradient ∂Z/∂E* of Z(E) = Σ|S' − E_i E_{i−m_k}|² (circular)."""
function gpa_gradient(E::AbstractVector, Sp::AbstractMatrix)
    N = length(E)
    m = delay_offsets(N)
    g = zeros(ComplexF64, N)
    @inbounds for k in 1:N
        mk = m[k]
        for i in 1:N
            j = mod1(i - mk, N)
            r = E[i] * E[j] - Sp[i, k]
            # ∂/∂E_i* of |r|² is r·conj(E_j); ∂/∂E_j* is r·conj(E_i)
            g[i] += r * conj(E[j])
            g[j] += r * conj(E[i])
        end
    end
    g
end

"""
Exact line search: the α ≥ 0 minimising Z(E − α d). With j = i − m_k,
r(α) = a + bα + cα², a = E_iE_j − S', b = −(d_iE_j + E_id_j), c = d_id_j.
"""
function gpa_linesearch(E::AbstractVector, d::AbstractVector, Sp::AbstractMatrix)
    N = length(E)
    m = delay_offsets(N)
    c0 = 0.0; c1 = 0.0; c2 = 0.0; c3 = 0.0; c4 = 0.0
    @inbounds for k in 1:N
        mk = m[k]
        for i in 1:N
            j = mod1(i - mk, N)
            a = E[i] * E[j] - Sp[i, k]
            b = -(d[i] * E[j] + E[i] * d[j])
            c = d[i] * d[j]
            c0 += abs2(a)
            c1 += 2real(conj(a) * b)
            c2 += abs2(b) + 2real(conj(a) * c)
            c3 += 2real(conj(b) * c)
            c4 += abs2(c)
        end
    end
    Z(α) = c0 + α * (c1 + α * (c2 + α * (c3 + α * c4)))
    # dZ/dα = c1 + 2c2 α + 3c3 α² + 4c4 α³ = 0
    cands = Float64[0.0]
    if c4 > 0
        p = [c1, 2c2, 3c3] ./ (4c4)              # monic cubic α³ + p3 α² + p2 α + p1
        C = [0.0 0.0 -p[1]; 1.0 0.0 -p[2]; 0.0 1.0 -p[3]]
        for r in eigvals(C)
            abs(imag(r)) < 1e-9 * max(1.0, abs(r)) && real(r) > 0 && push!(cands, real(r))
        end
    end
    cands[argmin(Z.(cands))]
end

"""
    retrieve_gpa(Imeas, dt; mask=nothing, opts=RetrievalOptions(),
                 nl=SHG(), callback=nothing) -> RetrievalResult

Generalized projections on an N×N trace `Imeas[ω, τ]`, same conventions and
callback as [`retrieve_pcgpa`](@ref).
"""
function retrieve_gpa(Imeas_in::AbstractMatrix, dt::Real; mask=nothing,
                      opts::RetrievalOptions=RetrievalOptions(),
                      nl::Nonlinearity=SHG(), callback=nothing)
    nl isa SHG || throw(ArgumentError("GPA is only implemented for SHG so far"))
    N = size(Imeas_in, 1)
    size(Imeas_in, 2) == N || throw(ArgumentError("GPA needs a square N×N trace"))
    grid = FourierGrid(N, dt)
    Imeas = normalize_trace(Imeas_in)
    rng = MersenneTwister(opts.seed)
    E = initial_guess(Imeas, grid, opts, rng)
    # GPA minimises against the projected field directly, so the overall
    # amplitude matters: start at the measured trace's scale.
    T0 = frog_trace(nl, E)
    E .*= (best_scale(Imeas, T0, mask))^(1 / 4)

    tr = Tracker(E)
    rep = Reporter(callback, opts)
    stopped = "max_iter"
    it = 0
    for outer it in 1:opts.max_iter
        Esig = signal_field(nl, E)
        G, Gp, μ = frog_error(Imeas, abs2.(Esig); mask=mask)
        record!(tr, E, G, Gp)
        if report!(rep, it, G, Gp, () -> μ .* abs2.(Esig), E)
            stopped = "user"; break
        end
        if G < opts.tol
            stopped = "tolerance"; break
        end
        # data constraint at μ = 1: the field itself carries the scale here
        intensity_projection!(Esig, Imeas, 1.0, mask)
        Sp = w2t(Esig)
        g = gpa_gradient(E, Sp)
        α = gpa_linesearch(E, g, Sp)
        α > 0 || (α = 1e-3 / max(maximum(abs.(g)), eps()))
        E = E .- α .* g
        E = recentre_integer(E)
    end
    res = finish(nl, tr, Imeas, grid, mask, it, stopped)
    report!(rep, it, res.G, res.Gp, () -> res.trace, res.E; force=true)
    res
end
