module SIWT
export  
    bestbasistree!,
    isiwpd,
    ShiftInvariantWaveletTransformNode,
    ShiftInvariantWaveletTransformObject,
    siwpd

using 
    Wavelets,
    Parameters,
    LinearAlgebra

using 
    ..Utils,
    ..DWT

import ..BestBasis: coefcost, ShannonEntropyCost

include("siwt/siwt_utls.jl")
include("siwt/siwt_one_level.jl")
include("siwt/siwt_bestbasis.jl")

"""
    siwpd(x, wt[, L, d])

Computes the Shift-Invariant Wavelet Packet Decomposition originally developed by Cohen, Raz
& Malah on the vector `x` using the discrete wavelet filter `wt` for `L` levels with depth
`d`.

# Arguments
- `x::AbstractVector{T} where T<:AbstractFloat`: 1D-signal.
- `wt::OrthoFilter`: Wavelet filter.
- `L::S where S<:Integer`: (Default: `maxtransformlevels(x)`) Number of transform levels.
- `d::S where S<:Integer`: (Default: `L`) Depth of shifted transform for each node.

# Returns
- `ShiftInvariantWaveletTransformObject` containing node and tree details.
"""
function siwpd(x::AbstractVector{T}, 
                wt::OrthoFilter, 
                L::S = maxtransformlevels(x),
                d::S = L) where {T<:AbstractFloat, S<:Integer}
    # Sanity check
    @assert 0 ≤ L ≤ maxtransformlevels(x)
    @assert 1 ≤ d ≤ L

    g, h = WT.makereverseqmfpair(wt, true)
    siwtObj = ShiftInvariantWaveletTransformObject(x, wt, L, d)
    rootNodeIndex = siwtObj.BestTree[1]
    siwpd_subtree!(siwtObj, rootNodeIndex, h, g, d)
    return siwtObj
end

function isiwpd(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂}) where
               {N, T₁<:Integer, T₂<:AbstractFloat}
    g, h = WT.makereverseqmfpair(siwtObj.Wavelet, true)
    rootNodeIndex = (0,0,0)
    isiwpd_subtree!(siwtObj, rootNodeIndex, h, g)

    return siwtObj.Nodes[rootNodeIndex].Value
end

end # end module