# The one entry point the Python side calls. Plain types in, plain types out,
# so the transport (juliacall now, a compiled app over stdin/stdout later) can
# change without the algorithms knowing.

const ALGORITHMS = ("pcgpa", "gpa", "ptychographic")

"""Build RetrievalOptions from a string-keyed Dict, ignoring unknown keys."""
function options_from_dict(d::AbstractDict)
    get_(k, default) = haskey(d, k) ? d[k] : default
    RetrievalOptions(
        max_iter       = Int(get_("max_iter", 500)),
        tol            = Float64(get_("tol", 1e-5)),
        snapshot_every = Int(get_("snapshot_every", 1)),
        min_snapshot_s = Float64(get_("min_snapshot_s", 0.1)),
        init           = Symbol(String(get_("init", "autocorrelation"))),
        init_fwhm_fs   = Float64(get_("init_fwhm_fs", 0.0)),
        seed           = Int(get_("seed", 0)),
        alpha_min      = Float64(get_("alpha_min", 0.1)),
        alpha_max      = Float64(get_("alpha_max", 0.5)),
        gamma          = Float64(get_("gamma", 0.0)),
    )
end

"""
    run_retrieval(algorithm, trace, dt; delays=nothing, mask=nothing,
                  options=Dict(), callback=nothing) -> Dict{String,Any}

* `algorithm` — "pcgpa" | "gpa" | "ptychographic"
* `trace` — I[ω, τ], rows frequency (offsets on FourierGrid(N, dt).w),
  columns delay. N×N for PCGPA/GPA.
* `dt` — time step of the grid, fs
* `delays` — measured delays (fs); required for "ptychographic" only
* `mask` — optional Bool matrix, true = measured pixel
* `options` — string-keyed Dict, see [`RetrievalOptions`](@ref)
* `callback(iter, G, Gp, trace, E)` — return `true` to stop

Returns a Dict with "E" (complex N-vector), "trace", "errors", "errors_p",
"G", "Gp",
"iterations", "stopped", "t" (fs) and "w" (rad/fs offsets).
"""
function run_retrieval(algorithm::AbstractString, trace::AbstractMatrix, dt::Real;
                       delays=nothing, mask=nothing, options::AbstractDict=Dict{String,Any}(),
                       callback=nothing)
    alg = lowercase(String(algorithm))
    opts = options_from_dict(options)
    I = Matrix{Float64}(trace)
    M = mask === nothing ? nothing : Matrix{Bool}(mask)
    res = if alg == "pcgpa"
        retrieve_pcgpa(I, dt; mask=M, opts=opts, callback=callback)
    elseif alg == "gpa"
        retrieve_gpa(I, dt; mask=M, opts=opts, callback=callback)
    elseif alg == "ptychographic"
        delays === nothing && throw(ArgumentError("ptychographic retrieval needs `delays`"))
        retrieve_ptychographic(I, Vector{Float64}(delays), dt; mask=M, opts=opts,
                               callback=callback)
    else
        throw(ArgumentError("unknown algorithm $algorithm; expected one of $(ALGORITHMS)"))
    end
    grid = FourierGrid(size(I, 1), dt)
    Dict{String,Any}(
        "E" => res.E, "trace" => res.trace, "errors" => res.errors, "errors_p" => res.errors_p,
        "G" => res.G, "Gp" => res.Gp,
        "iterations" => res.iterations, "stopped" => res.stopped,
        "t" => grid.t, "w" => grid.w,
    )
end

"""Simulated trace for the GUI and tests; returns a Dict like `run_retrieval`."""
function simulate(N::Integer, dt::Real; fwhm_fs=20.0, gdd_fs2=0.0, tod_fs3=0.0,
                  spm=0.0, noise=0.0, seed=0)
    T, E = simulate_trace(N, dt; fwhm_fs=fwhm_fs, gdd_fs2=gdd_fs2, tod_fs3=tod_fs3,
                          spm=spm, noise=noise, seed=seed)
    grid = FourierGrid(N, dt)
    Dict{String,Any}("trace" => T, "E" => E, "t" => grid.t, "w" => grid.w)
end
