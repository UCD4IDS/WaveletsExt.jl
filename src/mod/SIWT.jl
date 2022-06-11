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

@doc raw"""
    siwpd(x, wt[, L, d])

Computes the Shift-Invariant Wavelet Packet Decomposition originally developed by Cohen, Raz
& Malah on the vector `x` using the discrete wavelet filter `wt` for `L` levels with depth
`d`.

# Arguments
- `x::AbstractVector{T} where T<:AbstractFloat`: 1D-signal.
- `wt::OrthoFilter`: Wavelet filter.
- `L::S where S<:Integer`: (Default: `maxtransformlevels(x)`) Number of transform levels.
- `d::S where S<:Integer`: (Default: `L`) Depth of shifted transform for each node. Value of
  `d` must be strictly less than or equal to `L`.

# Returns
- `ShiftInvariantWaveletTransformObject` containing node and tree details.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SIWPD
siwpd(x, wt)        
siwpd(x, wt, 4)     # level 4 decomposition, where each decomposition has 4 levels of shifted decomposition.
siwpd(x, wt, 4, 2)  # level 4 decomposition, where each decomposition has 2 levels of shifted decomposition.
```

**See also:** [`isiwpd`](@ref), [`siwpd_subtree!`](@ref)
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

"""
    siwpd_subtree!(siwtObj, index, h, g, remainingRelativeDepth4ShiftedTransform[; signalNorm])

Runs the recursive computation of Shift-Invariant Wavelet Transform (SIWT) at each node
`index`. 

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where
  {N,T₁<:Integer,T₂<:AbstractFloat`: SIWT object.
- `index::NTuple{3,T₁} where T₁<:Integer`: Index of current node to be decomposed.
- `h::Vector{T₃} where T₃<:AbstractFloat`: High pass filter.
- `g::Vector{T₃} where T₃<:AbstractFloat`: Low pass filter.
- `remainingRelativeDepth4ShiftedTransform::T₁ where T₁<:Integer`: Remaining relative depth
  for shifted transform.

# Keyword Arguments
- `signalNorm::T₂ where T₂<:AbstractFloat`: (Default: `norm(siwtObj.Nodes[(0,0,0)].Value)`) 
  Signal Euclidean-norm.

**See also:** [`isiwpd_subtree!`](@ref)
"""
function siwpd_subtree!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                        index::NTuple{3,T₁},
                        h::Vector{T₃}, g::Vector{T₃},
                        remainingRelativeDepth4ShiftedTransform::T₁;
                        signalNorm::T₂ = norm(siwtObj.Nodes[(0,0,0)].Value)) where
                       {N, T₁<:Integer, T₂<:AbstractFloat, T₃<:AbstractFloat}
    treeMaxTransformLevel = siwtObj.MaxTransformLevel
    nodeDepth, _, nodeTransformShift = index

    @assert 0 ≤ nodeDepth ≤ treeMaxTransformLevel
    @assert 0 ≤ remainingRelativeDepth4ShiftedTransform ≤ treeMaxTransformLevel-nodeDepth

    # --- Base case ---
    isLeafNode = (nodeDepth == treeMaxTransformLevel)
    isShiftedTransform4NodeRequired = (remainingRelativeDepth4ShiftedTransform > 0)
    isShiftedTransformNode = nodeTransformShift > 0
    isShiftedTransformLeafNode = (!isShiftedTransform4NodeRequired && isShiftedTransformNode)
    if (isLeafNode || isShiftedTransformLeafNode)
        return nothing
    end
    
    #  --- General step ---
    #   - Decompose current node without additional shift 
    #   - Decompose children nodes
    childDepth = nodeDepth + 1
    (child1Index, child2Index) = sidwt_step!(siwtObj, index, h, g, false)
    childRemainingRelativeDepth4ShiftedTransform = isShiftedTransformNode ? 
        remainingRelativeDepth4ShiftedTransform-1 : 
        min(remainingRelativeDepth4ShiftedTransform, treeMaxTransformLevel-childDepth)
    siwpd_subtree!(siwtObj, child1Index, h, g, childRemainingRelativeDepth4ShiftedTransform, signalNorm=signalNorm)
    siwpd_subtree!(siwtObj, child2Index, h, g, childRemainingRelativeDepth4ShiftedTransform, signalNorm=signalNorm)

    # Case: remainingRelativeDepth4ShiftedTransform > 0
    #   - Decompose current node with additional shift
    #   - Decompose children (with additional shift) nodes
    if isShiftedTransform4NodeRequired
        (child1Index, child2Index) = sidwt_step!(siwtObj, index, h, g, true)
        childRemainingRelativeDepth4ShiftedTransform = remainingRelativeDepth4ShiftedTransform-1
        siwpd_subtree!(siwtObj, child1Index, h, g, childRemainingRelativeDepth4ShiftedTransform, signalNorm=signalNorm)
        siwpd_subtree!(siwtObj, child2Index, h, g, childRemainingRelativeDepth4ShiftedTransform, signalNorm=signalNorm)
    end

    return nothing
end

@doc raw"""
    isiwpd(siwtObj)

Computes the Inverse Shift-Invariant Wavelet Packet Decomposition originally developed by
Cohen, Raz & Malah.

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where
  {N,T₁<:Integer,T₂<:AbstractFloat`: SIWT object.

# Returns
- `Vector{T₂}`: Reconstructed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SIWPD
siwtObj = siwpd(x, wt)

# ISIWPD
isiwpd(siwtObj)
```

**See also:** [`siwpd`](@ref), [`isiwpd_subtree!`](@ref)
"""
function isiwpd(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂}) where
               {N, T₁<:Integer, T₂<:AbstractFloat}
    g, h = WT.makereverseqmfpair(siwtObj.Wavelet, true)
    rootNodeIndex = (0,0,0)
    isiwpd_subtree!(siwtObj, rootNodeIndex, h, g)

    return siwtObj.Nodes[rootNodeIndex].Value
end

"""
    isiwpd_subtree!(siwtObj, index, h, g)

Runs the recursive computation of Inverse Shift-Invariant Wavelet Transform (SIWT) at each
node `index`. 

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where
  {N,T₁<:Integer,T₂<:AbstractFloat`: SIWT object.
- `index::NTuple{3,T₁} where T₁<:Integer`: Index of current node to be decomposed.
- `h::Vector{T₃} where T₃<:AbstractFloat`: High pass filter.
- `g::Vector{T₃} where T₃<:AbstractFloat`: Low pass filter.

**See also:** [`siwpd_subtree!`](@ref)
"""
function isiwpd_subtree!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                         index::NTuple{3,T₁},
                         h::Vector{T₃}, g::Vector{T₃}) where
                        {N, T₁<:Integer, T₂<:AbstractFloat, T₃<:AbstractFloat}
    # Check for children nodes
    nodeDepth, nodeIndexAtDepth, nodeTransformShift = index
    hasNonShiftedChildren = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift) ∈ siwtObj.BestTree
    hasShiftedChildren = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift+(1<<nodeDepth)) ∈ siwtObj.BestTree

    # --- Base Case ---
    # If node has no children, return
    hasNoChildren = !(hasNonShiftedChildren || hasShiftedChildren)
    if hasNoChildren
        return nothing
    end

    # --- General Steps ---
    #   - If node has children, compute reconstruction from children nodes first
    #   - Once children nodes are reconstructed, compute reconstruction of current node
    #     based on coefficients of children nodes
    #   - After reconstruction of children nodes, delete children nodes
    #   - Return nothing
    @assert hasNonShiftedChildren ⊻ hasShiftedChildren
    childDepth = nodeDepth + 1
    child1IndexAtDepth = nodeIndexAtDepth<<1
    child2IndexAtDepth = nodeIndexAtDepth<<1 + 1
    childTransformShift = hasNonShiftedChildren ? nodeTransformShift : nodeTransformShift + (1<<nodeDepth)
    child1Index = (childDepth, child1IndexAtDepth, childTransformShift)
    child2Index = (childDepth, child2IndexAtDepth, childTransformShift)

    isiwpd_subtree!(siwtObj, child1Index, h, g)
    isiwpd_subtree!(siwtObj, child2Index, h, g)
    
    isidwt_step!(siwtObj, index, child1Index, child2Index, h, g)

    delete_node!(siwtObj, child1Index)
    delete_node!(siwtObj, child2Index)

    return nothing
end

end # end module