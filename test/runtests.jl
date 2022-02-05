ENV["GKSwstype"] = "100"
ENV["PLOTS_TEST"] = "true"

using 
    Test,
    Distributions,
    ImageQualityIndexes,
    Random,
    Plots,
    Statistics,
    Wavelets,
    WaveletsExt,
    SparseArrays

@testset "Utils" begin include("utils.jl") end
@testset "Transforms" begin include("transforms.jl") end
@testset "Wavelet Multiplication" begin include("wavemult.jl") end
@testset "Best Basis" begin include("bestbasis.jl") end
@testset "Denoising" begin include("denoising.jl") end
@testset "LDB" begin include("ldb.jl") end
@testset "Visualizations" begin include("visualizations.jl") end