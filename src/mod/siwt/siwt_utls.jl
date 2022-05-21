"""
    ShiftInvariantWaveletTransformObject{N, T₁<:Integer, T₂<:AbstractFloat}

Data structure to hold all shift-invariant wavelet transform (SIWT) information of a signal.

# Parameters
- `Nodes::Dict{NTuple{3,T₁}, Vector{ShiftInvariantWaveletTransformNode{N,T₁,T₂}}}`:
  Dictionary containing the information of each node within a tree.
- `SignalSize::Union{T₁, NTuple{N,T₁}}`: Size of the original signal.
- `MaxTransformLevel::T₁`: Maximum levels of transform set for signal.
- `Wavelet::OrthoFilter`: A discrete wavelet for transform purposes.
- `MinCost::Union{T₂, Nothing}`: The current minimum [`LogEnergyEntropyCost`](@ref) cost of 
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
    Nodes::Dict{NTuple{3,T₁}, Vector{ShiftInvariantWaveletTransformNode{N,T₁,T₂}}}
    SignalSize::Union{T₁, NTuple{N,T₁}}
    MaxTransformLevel::T₁
    Wavelet::OrthoFilter
    MinCost::Union{T₂, Nothing}
    BestTree::Vector{NTuple{3,T₁}}
end

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
- `Cost::T₂`: The [`LogEnergyEntropyCost`](@ref) of current node.
- `Value::Array{T₂,N}`: Coefficients of current node.
"""
mutable struct ShiftInvariantWaveletTransformNode{N, T₁<:Integer, T₂<:AbstractFloat}
    Depth::T₁
    IndexAtDepth::T₁
    TransformShift::T₁
    Cost::T₂
    Value::Array{T₂,N}

    function ShiftInvariantWaveletTransformNode{N, T₁, T₂}(Depth, IndexAtDepth, TransformShift, Cost, Value) where {N, T₁<:Integer, T₂<:AbstractFloat}
        valueDim = ndims(Value)
        if valueDim == 1
            maxIndexAtDepth = 1<<Depth - 1
            maxTransformShift = 1
        elseif valueDim == 2
            maxIndexAtDepth = 1<<(Depth<<1) - 1
            maxTransformShift = 3
        else
            (throw ∘ ArgumentError)("Coefficient array has dimension larger than 2.")
        end

        if (IndexAtDepth > maxIndexAtDepth) | (TransformShift > maxTransformShift)
            (throw ∘ ArgumentError)("Invalid IndexAtDepth or TransformShift for $(valueDim)D coefficients.")
        end
        return new(Depth, IndexAtDepth, TransformShift, Cost, Value)
    end
end

"""
    ShiftInvariantWaveletTransformObject(signal, wavelet)

Outer constructor and initialization of SIWT object.

# Arguments
- `signal::Array{T} where T<:AbstractFloat`: Input signal.
- `wavelet::OrthoFilter`: Wavelet filter.
"""
function ShiftInvariantWaveletTransformObject(signal::Array{T}, wavelet::OrthoFilter) where T<:AbstractFloat
    signalDim = ndims(signal)
    signalSize = signalDim == 1 ? length(signal) : size(signal)
    cost = coefcost(signal, LogEnergyEntropyCost())
    signalNode = ShiftInvariantWaveletTransformNode(0, 0, 0, cost, signal)
    maxTransformLevel = 0
    index = (signalNode.Depth, signalNode.IndexAtDepth, signalNode.TransformShift)
    S = typeof(signalNode.Depth)
    nodes = Dict{NTuple{3,S}, Vector{ShiftInvariantWaveletTransformNode{signalDim,S,T}}}(index => signal)
    tree = [index]
    return ShiftInvariantWaveletTransformObject(nodes, signalSize, maxTransformLevel, wavelet, cost, tree)
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
"""
function ShiftInvariantWaveletTransformNode(data::Array{T},
                                            depth::S,
                                            indexAtDepth::S,
                                            transformShift::S) where {T<:AbstractFloat, S<:Integer}
    cost = coefcost(data, LogEnergyEntropyCost())
    return ShiftInvariantWaveletTransformNode(depth, indexAtDepth, transformShift, cost, data)
end