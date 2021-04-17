__precompile__()

module WaveletsExt

include("mod/Utils.jl")
include("mod/WPD.jl")
include("mod/SIWPD.jl")
include("mod/BestBasis.jl")
include("mod/SWT.jl")
include("mod/Denoising.jl")
include("mod/LDB.jl")
include("mod/ACWT.jl")
include("mod/Visualizations.jl")

using Reexport
@reexport using .WPD,
                .BestBasis,
                .Denoising,
                .Utils,
                .LDB,
                .SWT,
                .SIWPD,
                .ACWT,
                .Visualizations

end