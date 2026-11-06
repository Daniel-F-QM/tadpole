# Build a Julia system image for the bundle (scripts/build_julia_bundle.py first).
#
#   julia --startup-file=no scripts/build_sysimage.jl
#
# Why: precompile caches are tied to where the depot lived when they were
# built — after the bundle is copied onto another machine, FFTW and
# TadpoleRetrieval were recompiled at first launch (~3.5 min). Code baked into
# a sysimage is not revalidated, so the moved bundle starts in seconds.
#
# PackageCompiler is installed into a throw-away tools environment
# (build/sysimage_tools), never into the bundle or the package.

const ROOT = dirname(@__DIR__)
const BUNDLE = joinpath(ROOT, "build", "julia_bundle")
const TOOLS = joinpath(ROOT, "build", "sysimage_tools")
const OUT = joinpath(BUNDLE, "sysimage", Sys.iswindows() ? "tadpole.dll" : "tadpole.so")

isdir(BUNDLE) || error("no bundle at $BUNDLE — run scripts/build_julia_bundle.py first")
bundle_julia = joinpath(BUNDLE, "runtime", "bin", Sys.iswindows() ? "julia.exe" : "julia")
# The sysimage must match the runtime that will load it.
bundle_version = VersionNumber(readchomp(`$bundle_julia --startup-file=no -e "print(VERSION)"`))
bundle_version == VERSION ||
    error("run this with the bundle's Julia ($bundle_version), not $VERSION")

using Pkg
Pkg.activate(TOOLS)
Base.find_package("PackageCompiler") === nothing && Pkg.add("PackageCompiler")
using PackageCompiler

# Resolve the bundle env's packages from the bundle depot. APPENDED, not
# prepended: the first depot receives new downloads, and PackageCompiler's
# MinGW compiler (~0.7 GB) must land in the build machine's depot, not in the
# bundle that ships.
push!(DEPOT_PATH, joinpath(BUNDLE, "depot"))
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

workload = joinpath(TOOLS, "workload.jl")
write(workload, """
using TadpoleRetrieval
d = TadpoleRetrieval.simulate(64, 4.0; fwhm_fs=20.0, gdd_fs2=100.0)
for alg in ("pcgpa", "gpa")
    run_retrieval(alg, d["trace"], 4.0;
                  options=Dict{String,Any}("max_iter" => 5, "min_snapshot_s" => 0.0),
                  callback=(it, G, Gp, T, E) -> false)
end
""")

mkpath(dirname(OUT))
create_sysimage(["TadpoleRetrieval", "PythonCall"];
                project=joinpath(BUNDLE, "env"),
                sysimage_path=OUT,
                precompile_execution_file=workload)
println("sysimage written: $OUT ($(round(filesize(OUT) / 2^20; digits=1)) MB)")
