module WaveMult
export mat2sparse_nsform,
       ns_dwt
       
using Wavelets,
      LinearAlgebra, 
      SparseArrays

import ..DWT: dwt_step!, idwt_step!

include("wavemult/utils.jl")
include("wavemult/mat2sparse.jl")
include("wavemult/wavemult.jl")
include("wavemult/transforms.jl")
end # module