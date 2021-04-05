using 
    Test,
    Distributions,
    Random,
    Plots,
    Statistics,
    Wavelets,
    WaveletsExt

@testset "Utils" begin include("utils.jl") end
@testset "Visualizations" begin include("visualizations.jl") end
@testset "Transforms" begin include("transforms.jl") end
@testset "Denoising" begin include("denoising.jl") end
@testset "Best Basis" begin include("bestbasis.jl") end
@testset "LDB" begin include("ldb.jl") end