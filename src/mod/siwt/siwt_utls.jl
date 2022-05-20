mutable struct ShiftInvariantWaveletTransformObject{N₁, N₂, T₁<:Integer, T₂<:AbstractFloat}
    Nodes::Dict{NTuple{N₁,T}, Vector{ShiftInvariantWaveletTransformNode{N₂,T₁,T₂}}}
    SignalLength::T₁
    MaxTransformLevel::T₁
    Wavelet::OrthoFilter
    MinCost::Union{T₂, Nothing}
    Tree::Union{Vector{NTuple{N₁,T₁}}, Nothing}
end

struct ShiftInvariantWaveletTransformNode{N, T₁<:Integer, T₂<:AbstractFloat}
    IsShiftedTransform::Bool
    Depth::T₁
    IndexAtDepth::T₁
    IndexAtTree::T₁
    Cost::Union{T₂, Nothing}
    Value::Array{T₂,N}
end

# TODO: Create constructors