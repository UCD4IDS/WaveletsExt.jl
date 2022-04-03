module WaveMult
export mat2sparseform_std,
       mat2sparseform_nonstd,
       ns_dwt,
       ns_idwt,
       std_wavemult,
       nonstd_wavemult
       
using Wavelets,
      LinearAlgebra, 
      SparseArrays

import ..DWT: dwt_step!, idwt_step!

include("wavemult/utils.jl")
include("wavemult/mat2sparse.jl")
include("wavemult/wavemult.jl")
include("wavemult/transforms.jl")
end # module