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

@testset verbose=true "Utils" begin include("utils.jl") end
@testset verbose=true "Transforms" begin include("transforms.jl") end
@testset verbose=true "Wavelet Multiplication" begin include("wavemult.jl") end
@testset verbose=true "Best Basis" begin include("bestbasis.jl") end
@testset verbose=true "Denoising" begin include("denoising.jl") end
@testset verbose=true "LDB" begin include("ldb.jl") end
@testset verbose=true "Visualizations" begin include("visualizations.jl") end