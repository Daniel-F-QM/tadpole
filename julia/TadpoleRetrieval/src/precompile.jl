# Run a tiny version of every entry point at precompile time so the first
# real call from the GUI does not pay for compilation.

@setup_workload begin
    @compile_workload begin
        d = simulate(32, 4.0; fwhm_fs=20.0, gdd_fs2=50.0)
        cb = (it, G, Gp, T, E) -> false
        for alg in ("pcgpa", "gpa")
            run_retrieval(alg, d["trace"], 4.0;
                          options=Dict{String,Any}("max_iter" => 3, "min_snapshot_s" => 0.0),
                          callback=cb)
        end
    end
end
