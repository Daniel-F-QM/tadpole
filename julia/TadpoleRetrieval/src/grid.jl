# The Fourier grid shared by the time and frequency domains.
#
# N points, t_k = (k - N/2 - 1)·dt for k = 1…N (so t = 0 sits at index N/2+1),
# and the matching frequency offsets ω_k = (k - N/2 - 1)·dω with
# dω = 2π / (N·dt). Δt·Δν = 1/N is what makes the grid "Fourier consistent".

"""
    FourierGrid(N, dt)

A centred time/frequency grid with `N` points and time step `dt` (fs).
"""
struct FourierGrid
    N::Int
    dt::Float64
    dw::Float64
    t::Vector{Float64}
    w::Vector{Float64}
end

function FourierGrid(N::Integer, dt::Real)
    N >= 4 || throw(ArgumentError("grid needs at least 4 points, got $N"))
    iseven(N) || throw(ArgumentError("grid size must be even, got $N"))
    dt > 0 || throw(ArgumentError("dt must be positive, got $dt"))
    dw = 2π / (N * dt)
    k = collect(0:N-1) .- N ÷ 2
    FourierGrid(N, Float64(dt), dw, k .* dt, k .* dw)
end

# Index offset of delay column k: τ_k = m_k · dt.
delay_offsets(N::Integer) = collect(0:N-1) .- N ÷ 2

"""Time → frequency along dim 1 (kernel e^{-iωt}), centred both sides."""
t2w(x::AbstractArray) = fftshift(fft(ifftshift(x, 1), 1), 1)

"""Frequency → time along dim 1, the exact inverse of `t2w`."""
w2t(x::AbstractArray) = fftshift(ifft(ifftshift(x, 1), 1), 1)

"""
    shift_field(E, grid, τ)

E(t − τ) for an arbitrary (non-integer) τ via the Fourier shift theorem.
Used to centre pulses; the ptychographic algorithm will need it for
non-uniform delays.
"""
function shift_field(E::AbstractVector, grid::FourierGrid, τ::Real)
    Ew = t2w(E)
    w2t(Ew .* exp.(-im .* grid.w .* τ))
end
