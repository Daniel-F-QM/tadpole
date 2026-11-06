using Test
using TadpoleRetrieval
const TR = TadpoleRetrieval

@testset "TadpoleRetrieval" begin
    @testset "grid" begin
        g = FourierGrid(128, 2.0)
        @test g.dt * g.dw ≈ 2π / 128
        @test g.t[65] == 0.0 && g.w[65] == 0.0
        x = randn(ComplexF64, 128)
        @test TR.w2t(TR.t2w(x)) ≈ x
        # Fourier shift agrees with an integer circshift
        @test TR.shift_field(x, g, 3 * g.dt) ≈ circshift(x, 3)
        @test_throws ArgumentError FourierGrid(127, 1.0)
    end

    @testset "trace + simulate" begin
        T, E = TR.simulate_trace(64, 4.0; fwhm_fs=20.0, gdd_fs2=100.0)
        @test size(T) == (64, 64)
        @test maximum(T) ≈ 1.0
        # SHG traces are symmetric in delay (column N/2+1 is τ = 0)
        @test T[:, 34] ≈ T[:, 32] atol = 1e-10
        @test pulse_distance(E, E, FourierGrid(64, 4.0)) < 1e-6
        G, Gp, μ = frog_error(T, 2 .* T)
        @test G < 1e-12 && Gp < 1e-12 && μ ≈ 0.5
        # G′ is G rescaled by the trace's own norm: sqrt(n / ΣI²)
        Tn = T .+ 0.01
        G, Gp, μ = frog_error(T, Tn)
        @test Gp ≈ G * sqrt(length(T) / sum(abs2, T)) rtol = 1e-10
    end

    @testset "outer-product index is a permutation" begin
        idx = TR.outer_index(16)
        @test sort(vec(idx)) == collect(1:256)
        E = randn(ComplexF64, 16)
        O = E * transpose(E)
        @test O[idx] ≈ TR.signal_matrix(SHG(), E)
    end

    @testset "GPA line search hits the quartic minimum" begin
        E = randn(ComplexF64, 16); d = randn(ComplexF64, 16)
        Sp = randn(ComplexF64, 16, 16)
        Z(α) = sum(abs2, TR.signal_matrix(SHG(), E .- α .* d) .- Sp)
        α = TR.gpa_linesearch(E, d, Sp)
        @test Z(α) <= Z(α * 0.99) + 1e-9 && Z(α) <= Z(α * 1.01) + 1e-9
    end

    @testset "$alg retrieves a chirped pulse" for alg in ("pcgpa", "gpa")
        N, dt = 64, 4.0
        T, E = TR.simulate_trace(N, dt; fwhm_fs=20.0, gdd_fs2=200.0)
        calls = Ref(0)
        cb = (it, G, Gp, Tr, Ef) -> (calls[] += 1; size(Tr) == (N, N) || error("bad"); false)
        d = run_retrieval(alg, T, dt;
                          options=Dict{String,Any}("max_iter" => 500, "tol" => 1e-6,
                                                   "min_snapshot_s" => 0.0),
                          callback=cb)
        @test d["G"] < 1e-3 && d["Gp"] < 1e-2
        @test length(d["errors_p"]) == d["iterations"]
        @test pulse_distance(d["E"], E, FourierGrid(N, dt)) < 0.05
        @test calls[] > 1
        @test length(d["errors"]) == d["iterations"]
    end

    @testset "$alg reaches the noise floor" for alg in ("pcgpa", "gpa")
        σ = 0.003
        T, E = TR.simulate_trace(64, 4.0; fwhm_fs=20.0, gdd_fs2=200.0, noise=σ, seed=1)
        d = run_retrieval(alg, T, 4.0; options=Dict{String,Any}("max_iter" => 300))
        # rms residual can't beat the noise, but should come close to it
        @test d["G"] < 1.5σ
        @test pulse_distance(d["E"], E, FourierGrid(64, 4.0)) < 0.15
    end

    @testset "callback can stop a run" begin
        T, _ = TR.simulate_trace(64, 4.0; gdd_fs2=200.0)
        d = run_retrieval("gpa", T, 4.0;
                          options=Dict{String,Any}("max_iter" => 1000, "tol" => 0.0,
                                                   "min_snapshot_s" => 0.0),
                          callback=(it, G, Gp, Tr, E) -> it >= 5)
        @test d["stopped"] == "user" && d["iterations"] == 5
    end

    @testset "ptychographic skeleton" begin
        T, _ = TR.simulate_trace(32, 4.0)
        @test_throws ErrorException run_retrieval("ptychographic", T, 4.0;
                                                  delays=collect(1.0:32.0))
        @test_throws ArgumentError run_retrieval("ptychographic", T, 4.0)
        @test TR.soft_threshold(0.5, 1.0) == 0.0
        @test TR.soft_threshold(-3.0, 1.0) == -2.0
    end
end
