# The FROG geometry, as a type. Every algorithm asks the nonlinearity for its
# signal field, so adding THG / PG / SD later means one new subtype with a
# `gate` method (and, for PCGPA, an outer-product update rule) — the
# algorithms themselves stay unchanged.

abstract type Nonlinearity end

"""Second-harmonic FROG: the gate is the pulse itself."""
struct SHG <: Nonlinearity end

"""The gate field G(t) for pulse E(t)."""
gate(::SHG, E::AbstractVector) = E

"""
    signal_matrix(nl, E, G)

S[i, k] = E(t_i) · G(t_i − τ_k) with τ_k = m_k·dt on the same grid, circular
indexing. Columns are delays, rows are time.
"""
function signal_matrix(::Nonlinearity, E::AbstractVector, G::AbstractVector)
    N = length(E)
    m = delay_offsets(N)
    S = Matrix{ComplexF64}(undef, N, N)
    @inbounds for k in 1:N, i in 1:N
        S[i, k] = E[i] * G[mod1(i - m[k], N)]
    end
    S
end

signal_matrix(nl::Nonlinearity, E::AbstractVector) = signal_matrix(nl, E, gate(nl, E))

"""Complex signal field in the frequency domain, E_sig(ω, τ)."""
signal_field(nl::Nonlinearity, E::AbstractVector) = t2w(signal_matrix(nl, E))

"""The FROG trace |E_sig(ω, τ)|² of pulse `E`."""
frog_trace(nl::Nonlinearity, E::AbstractVector) = abs2.(signal_field(nl, E))
frog_trace(E::AbstractVector) = frog_trace(SHG(), E)
