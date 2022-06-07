__precompile__()

module WaveletsExt

include("mod/Utils.jl")
include("mod/DWT.jl")
include("mod/ACWT.jl")
include("mod/BestBasis.jl")
include("mod/SIWT.jl")
include("mod/SWT.jl")
include("mod/Denoising.jl")
include("mod/LDB.jl")
include("mod/Visualizations.jl")
include("mod/WaveMult.jl")

using Reexport
@reexport using .DWT,
                .BestBasis,
                .Denoising,
                .Utils,
                .LDB,
                .SWT,
                .SIWT,
                .ACWT,
                .Visualizations,
                .WaveMult

end