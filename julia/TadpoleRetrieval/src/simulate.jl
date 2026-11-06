# Test pulses with a known answer, for the test suite and the GUI's
# "Simulate" button.

"""
    simulate_pulse(grid; fwhm_fs=20, gdd_fs2=0, tod_fs3=0, spm=0, rng=nothing, noise=0)

A Gaussian pulse with transform-limited intensity FWHM `fwhm_fs`, given a
spectral phase φ(ω) = GDD/2·ω² + TOD/6·ω³, and optionally a time-domain
self-phase-modulation term `spm`·|E(t)|² (rad at the peak) to make a pulse
with structure. Returned centred, peak |E| = 1.
"""
function simulate_pulse(grid::FourierGrid; fwhm_fs::Real=20.0, gdd_fs2::Real=0.0,
                        tod_fs3::Real=0.0, spm::Real=0.0)
    E0 = exp.(-2log(2) .* (grid.t ./ fwhm_fs) .^ 2) .+ 0im
    Ew = t2w(E0)
    w = grid.w
    Ew .*= exp.(im .* (gdd_fs2 / 2 .* w .^ 2 .+ tod_fs3 / 6 .* w .^ 3))
    E = w2t(Ew)
    if spm != 0
        E .*= exp.(im .* spm .* abs2.(E) ./ maximum(abs2.(E)))
    end
    center_pulse(E, grid)
end

"""
    simulate_trace(N, dt; noise=0, seed=0, kwargs...) -> (trace, E)

SHG trace of `simulate_pulse` on an N×N grid, normalised to peak 1, with
additive Gaussian noise of standard deviation `noise` (relative to the peak).
"""
function simulate_trace(N::Integer, dt::Real; noise::Real=0.0, seed::Integer=0, kwargs...)
    grid = FourierGrid(N, dt)
    E = simulate_pulse(grid; kwargs...)
    T = frog_trace(SHG(), E)
    T ./= maximum(T)
    if noise > 0
        rng = MersenneTwister(seed)
        T .+= noise .* randn(rng, size(T))
    end
    T, E
end
