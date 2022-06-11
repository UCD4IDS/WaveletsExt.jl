"""
    ShiftInvariantWaveletTransformNode{N, T₁<:Integer, T₂<:AbstractFloat}

Data structure to hold the index, coefficients, and cost value of an SIWT node.

# Parameters
- `Depth::T₁`: The depth of current node. Root node has depth 0.
- `IndexAtDepth::T₁`: The node index at current depth. Index starts from 0 at each depth.
- `TransformShift::T₁`: The type of shift operated on the parent node before computing the
  transform. Accepted type of shift values are:
    - `0`: Coefficients of parent node are not shifted prior to transform.
    - `1`: For both 1D and 2D signals, coefficients of parent node are circularly shifted to 
      the left by 1 index.
    - `2`: For 2D signals, coefficients of parent node are circularly shifted from bottom to
      top by 1 index. Not available for 1D signals.
    - `3`: For 2D signals, coefficients of parent node are circularly shifted from bottom to
      top and right to left by 1 index each. Not available for 1D signals.
- `Cost::T₂`: The [`ShannonEntropyCost`](@ref) of current node.
- `Value::Array{T₂,N}`: Coefficients of current node.

**See also:** [`ShiftInvariantWaveletTransformObject`](@ref)
"""
mutable struct ShiftInvariantWaveletTransformNode{N, T₁<:Integer, T₂<:AbstractFloat}
    Depth::T₁
    IndexAtDepth::T₁
    TransformShift::T₁
    Cost::T₂
    Value::Array{T₂,N}

    function ShiftInvariantWaveletTransformNode{N, T₁, T₂}(Depth, IndexAtDepth, TransformShift, Cost, Value) where {N, T₁<:Integer, T₂<:AbstractFloat}
        if N ≠ ndims(Value)
            (throw ∘ TypeError)("Value array is not of $N dimension.")
        end

        if N == 1
            maxIndexAtDepth = 1<<Depth - 1
            maxTransformShift = maxIndexAtDepth
        elseif N == 2
            maxIndexAtDepth = 1<<(Depth<<1) - 1
            maxTransformShift = 3
            (throw ∘ ArgumentError)("2D SIWT not available yet.")
        else
            (throw ∘ ArgumentError)("Coefficient array has dimension larger than 2.")
        end

        if (IndexAtDepth > maxIndexAtDepth) | (TransformShift > maxTransformShift)
            (throw ∘ ArgumentError)("Invalid IndexAtDepth or TransformShift for $(N)D coefficients.")
        end
        return new(Depth, IndexAtDepth, TransformShift, Cost, Value)
    end
end

"""
    ShiftInvariantWaveletTransformObject{N, T₁<:Integer, T₂<:AbstractFloat}

Data structure to hold all shift-invariant wavelet transform (SIWT) information of a signal.

# Parameters
- `Nodes::Dict{NTuple{3,T₁}, ShiftInvariantWaveletTransformNode{N,T₁,T₂}}`:
  Dictionary containing the information of each node within a tree.
- `SignalSize::Union{T₁, NTuple{N,T₁}}`: Size of the original signal.
- `MaxTransformLevel::T₁`: Maximum levels of transform set for signal.
- `Wavelet::OrthoFilter`: A discrete wavelet for transform purposes.
- `MinCost::Union{T₂, Nothing}`: The current minimum [`ShannonEntropyCost`](@ref) cost of 
  the decomposition tree.

!!! note 
    `MinCost` parameter will contain the cost of the root node by default. To compute the 
    true minimum cost of the decomposition tree, one will need to first compute the SIWT
    best basis by calling [`bestbasistree`](@ref).

- `BestTree::Vector{NTuple{3,T₁}}`: A collection of node indices that belong in the best
  basis tree.

!!! note 
    `BestTree` parameter will contain all the nodes by default, ie. the full decomposition 
    will be the best tree. To get the SIWT best basis tree that produce the minimum cost, 
    one will neeed to call [`bestbasistree`](@ref).

**See also:** [`ShiftInvariantWaveletTransformNode`](@ref)
"""
mutable struct ShiftInvariantWaveletTransformObject{N, T₁<:Integer, T₂<:AbstractFloat}
    Nodes::Dict{NTuple{3,T₁}, ShiftInvariantWaveletTransformNode{N,T₁,T₂}}
    SignalSize::Union{T₁, NTuple{N,T₁}}
    MaxTransformLevel::T₁
    MaxShiftedTransformLevels::T₁
    Wavelet::OrthoFilter
    MinCost::Union{T₂, Nothing}
    BestTree::Vector{NTuple{3,T₁}}

    function ShiftInvariantWaveletTransformObject{N,T₁,T₂}(nodes, signalSize, maxTransformLevel, maxShiftedTransformLevels, wt, minCost, bestTree) where {N, T₁<:Integer, T₂<:AbstractFloat}
        0 ≤ maxTransformLevel ≤ maxtransformlevels(nodes[(0,0,0)].Value) || (throw ∘ ArgumentError)("Provided MaxTransformLevels is too large.")
        0 ≤ maxShiftedTransformLevels < length(nodes[(0,0,0)].Value) || (throw ∘ ArgumentError)("Provided MaxShiftedTransformLevels is too large.")
        return new(nodes, signalSize, maxTransformLevel, maxShiftedTransformLevels, wt, minCost, bestTree)
    end
end

"""
    ShiftInvariantWaveletTransformNode(data, depth, indexAtDepth, transformShift)

Outer constructor of SIWT node.

# Arguments
- `data::Array{T} where T<:AbstractFloat`: Array of coefficients.
- `depth::S where S<:Integer`: Depth of current node.
- `indexAtDepth::S where S<:Integer`: Node index at current depth.
- `transformShift::S where S<:Integer`: The type of shift operated on the parent node before
  computing the transform.
- `nrm::T where T<:AbstractFloat`: (Default: `norm(data)`) Norm of the signal.
"""
function ShiftInvariantWaveletTransformNode(data::Array{T},
                                            depth::S,
                                            indexAtDepth::S,
                                            transformShift::S,
                                            nrm::T = norm(data)) where {T<:AbstractFloat, S<:Integer}
    cost = coefcost(data, ShannonEntropyCost(), nrm)
    N = ndims(data)
    return ShiftInvariantWaveletTransformNode{N,S,T}(depth, indexAtDepth, transformShift, cost, data)
end

"""
    ShiftInvariantWaveletTransformObject(signal, wavelet)

Outer constructor and initialization of SIWT object.

# Arguments
- `signal::Array{T} where T<:AbstractFloat`: Input signal.
- `wavelet::OrthoFilter`: Wavelet filter.
- `maxTransformLevel::S where S<:Integer`: (Default: `0`) Max transform level.
- `maxShiftedTransformLevel::S where S<:Integer`: (Default: `0`) Max shifted transform
  levels.
"""
function ShiftInvariantWaveletTransformObject(signal::Array{T}, 
                                              wavelet::OrthoFilter,
                                              maxTransformLevel::S = 0,
                                              maxShiftedTransformLevel::S = 0) where 
                                             {T<:AbstractFloat, S<:Integer}
    signalDim = ndims(signal)
    signalSize = signalDim == 1 ? length(signal) : size(signal)
    cost = coefcost(signal, ShannonEntropyCost())
    signalNode = ShiftInvariantWaveletTransformNode{signalDim,S,T}(0, 0, 0, cost, signal)
    index = (signalNode.Depth, signalNode.IndexAtDepth, signalNode.TransformShift)
    nodes = Dict{NTuple{3,S}, ShiftInvariantWaveletTransformNode{signalDim,S,T}}(index => signalNode)
    tree = [index]
    return ShiftInvariantWaveletTransformObject{signalDim,S,T}(nodes, signalSize, maxTransformLevel, maxShiftedTransformLevel, wavelet, cost, tree)
end

"""
    Wavelets.Util.isvalidtree(siwtObj)

Checks if tree within SIWT object is a valid tree.

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject`: SIWT object.

# Returns
- `bool`: `true` if tree is valid, `false` otherwise.

!!! note
    `isvalidtree` checks the criteria that each node has a parent (except root node) and one
    set of children. A node can have either non-shifted or shifted children, but not both.
    Using this function on a decomposed `siwtObj` prior to its best basis search will return
    `false`. 

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SIWPD
siwtObj = siwpd(x, wt)
isvalidtree(siwtObj)        # Returns false

# Best basis tree
bestbasistree!(siwtObj)
isvalidtree(siwtObj)        # Returns true
```
"""
function Wavelets.Util.isvalidtree(siwtObj::ShiftInvariantWaveletTransformObject)
    @assert (Set ∘ keys)(siwtObj.Nodes) == Set(siwtObj.BestTree)

    # Each node needs to:
    #   - Be a root node OR have a parent node
    #   - Have child nodes XOR have shifted child nodes XOR have no child nodes
    nodeSet = Set(siwtObj.BestTree)
    for index in nodeSet
        depth, indexAtDepth, transformShift = index

        isRootNode = index == (0,0,0)
        hasParentNode = (depth-1, indexAtDepth>>1, transformShift) ∈ nodeSet

        hasChildNodes = (depth+1, indexAtDepth<<1, transformShift) ∈ nodeSet && (depth+1, indexAtDepth<<1+1, transformShift) ∈ nodeSet
        hasShiftedChildNodes = (depth+1, indexAtDepth<<1, transformShift+(1<<depth)) ∈ nodeSet && (depth+1, indexAtDepth<<1+1, transformShift+(1<<depth)) ∈ nodeSet
        isLeafNode = !hasChildNodes && !hasShiftedChildNodes

        parentCheck = isRootNode ⊻ hasParentNode
        childCheck = isLeafNode ⊻ hasChildNodes ⊻ hasShiftedChildNodes

        if !(parentCheck && childCheck)
            return false
        end
    end
    return true
end

"""
    delete_node!(siwtObj, index)

Deletes a node and all its children from an SIWT object.

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where {N, T₁<:Integer,
  T₂<:AbstractFloat}`: SIWT object.
- `index::NTuple{3,T₁} where T₁<:Integer`: Index of node to be deleted.
"""
function delete_node!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                      index::NTuple{3,T₁}) where
                     {N, T₁<:Integer, T₂<:AbstractFloat}
    # Node unavailable
    if index ∉ siwtObj.BestTree
        return nothing
    end

    # Delete node
    delete!(siwtObj.Nodes, index)
    deleteat!(siwtObj.BestTree, findall(x -> x==index, siwtObj.BestTree))

    # Delete children nodes
    nodeDepth, nodeIndexAtDepth, nodeTransformShift = index
    child1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift)
    child2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift)
    shiftedChild1Index = (nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift+(1<<nodeDepth))
    shiftedChild2Index = (nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift+(1<<nodeDepth))
    delete_node!(siwtObj, child1Index)
    delete_node!(siwtObj, child2Index)
    delete_node!(siwtObj, shiftedChild1Index)
    delete_node!(siwtObj, shiftedChild2Index)
    
    return nothing
end