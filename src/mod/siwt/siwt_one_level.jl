"""
    sidwt_step!(siwtObj, index, h, g, shiftedTransform[; signalNorm])

Computes one step of the SIWT decomposition on the node `index`.

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where
  {N,T₁<:Integer,T₂<:AbstractFloat`: SIWT object.
- `index::NTuple{3,T₁} where T₁<:Integer`: Index of current node to be decomposed.
- `h::Vector{T₃} where T₃<:AbstractFloat`: High pass filter.
- `g::Vector{T₃} where T₃<:AbstractFloat`: Low pass filter.
- `shiftedTransform::Bool`: Whether a shifted transform should be performed.

# Keyword Arguments
- `signalNorm::T₂ where T₂<:AbstractFloat`: (Default: `norm(siwtObj.Nodes[(0,0,0)].Value`)
  Signal Euclidean-norm.

# Returns
- `child1Index::NTuple{3,T₁}`: Child 1 index.
- `child2Index::NTuple{3,T₁}`: Child 2 index.
"""
function sidwt_step!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                     index::NTuple{3,T₁},
                     h::Vector{T₃}, g::Vector{T₃},
                     shiftedTransform::Bool;
                     signalNorm::T₂ = norm(siwtObj.Nodes[(0,0,0)].Value)) where 
                    {N, T₁<:Integer, T₂<:AbstractFloat, T₃<:AbstractFloat}
    nodeObj = siwtObj.Nodes[index]
    nodeValue = nodeObj.Value
    nodeLength = length(nodeValue)
    nodeDepth, nodeIndexAtDepth, nodeTransformShift = index

    childLength = nodeLength ÷ 2
    child1Value = Vector{T₂}(undef, childLength)
    child2Value = Vector{T₂}(undef, childLength)
    sidwt_step!(child1Value, child2Value, nodeValue, h, g, shiftedTransform)
    child1Obj = ShiftInvariantWaveletTransformNode(child1Value, nodeDepth+1, nodeIndexAtDepth<<1, nodeTransformShift+(1<<nodeDepth)*shiftedTransform, signalNorm)
    child2Obj = ShiftInvariantWaveletTransformNode(child2Value, nodeDepth+1, nodeIndexAtDepth<<1+1, nodeTransformShift+(1<<nodeDepth)*shiftedTransform, signalNorm)
    siwtObj.Nodes[(child1Obj.Depth, child1Obj.IndexAtDepth, child1Obj.TransformShift)] = child1Obj
    siwtObj.Nodes[(child2Obj.Depth, child2Obj.IndexAtDepth, child2Obj.TransformShift)] = child2Obj

    child1Index = (child1Obj.Depth, child1Obj.IndexAtDepth, child1Obj.TransformShift)
    child2Index = (child2Obj.Depth, child2Obj.IndexAtDepth, child2Obj.TransformShift)
    push!(siwtObj.BestTree, child1Index)
    push!(siwtObj.BestTree, child2Index)

    return (child1Index, child2Index)
end

"""
    sidwt_step!(w₁, w₂, v, h, g, s)

Computes one step of the SIWT decomposition on the node `v`.

# Arguments
- `w₁::AbstractVector{T} where T<:AbstractFloat`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:AbstractFloat`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:AbstractFloat`: Vector of coefficients from a node at level `d`.
- `h::Vector{S} where S<:AbstractFloat`: High pass filter.
- `g::Vector{S} where S<:AbstractFloat`: Low pass filter.
- `s::Bool`: Whether a shifted transform should be performed.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.
"""
function sidwt_step!(w₁::AbstractVector{T}, w₂::AbstractVector{T},
                     v::AbstractVector{T},
                     h::Vector{S}, g::Vector{S},
                     s::Bool) where {T<:AbstractFloat, S<:AbstractFloat}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)÷2
    @assert length(h) == length(g)

    # Setup
    n = length(v)           # Parent length
    n₁ = length(w₁)         # Child length
    filtlen = length(h)     # Filter length

    # One step of discrete transform
    for i in 1:n₁
        k₁ = mod1(2*i-1-s, n)   # Start index for low pass filtering
        k₂ = 2*i-s              # Start index for high pass filtering
        @inbounds w₁[i] = g[end] * v[k₁]
        @inbounds w₂[i] = h[1] * v[k₂]
        for j in 2:filtlen
            k₁ = k₁+1 |> k₁ -> k₁>n ? mod1(k₁,n) : k₁
            k₂ = k₂-1 |> k₂ -> k₂≤0 ? mod1(k₂,n) : k₂
            @inbounds w₁[i] += g[end-j+1] * v[k₁]
            @inbounds w₂[i] += h[j] * v[k₂]
        end
    end
    return w₁, w₂
end

"""
    isidwt_step!(siwtObj, index, child1Index, child2Index, h, g)

Computes one step of the inverse SIWT decomposition on the node `index`.

# Arguments
- `siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂} where
  {N,T₁<:Integer,T₂<:AbstractFloat`: SIWT object.
- `index::NTuple{3,T₁} where T₁<:Integer`: Index of current node to be decomposed.
- `child1Index::NTuple{T,T₁} where T₁<:Integer`: Index of child 1.
- `child2Index::NTuple{T,T₁} where T₁<:Integer`: Index of child 2.
- `h::Vector{T₃} where T₃<:AbstractFloat`: High pass filter.
- `g::Vector{T₃} where T₃<:AbstractFloat`: Low pass filter.
"""
function isidwt_step!(siwtObj::ShiftInvariantWaveletTransformObject{N,T₁,T₂},
                      nodeIndex::NTuple{3,T₁},
                      child1Index::NTuple{3,T₁}, child2Index::NTuple{3,T₁},
                      h::Vector{T₃}, g::Vector{T₃}) where
                     {N, T₁<:Integer, T₂<:AbstractFloat, T₃<:AbstractFloat}
    nodeObj = siwtObj.Nodes[nodeIndex]
    child1Obj = siwtObj.Nodes[child1Index]
    child2Obj = siwtObj.Nodes[child2Index]

    @assert child1Obj.TransformShift == child2Obj.TransformShift
    isShiftedTransform = nodeObj.TransformShift == child1Obj.TransformShift

    nodeValue = nodeObj.Value
    child1Value = child1Obj.Value
    child2Value = child2Obj.Value
    isidwt_step!(nodeValue, child1Value, child2Value, h, g, isShiftedTransform)

    return nothing
end

"""
    isidwt_step!(v, w₁, w₂, h, g, s)

Computes one step of the inverse SIWT decomposition on the node `w₁` and `w₂`.

# Arguments
- `v::AbstractVector{T} where T<:AbstractFloat`: Vector allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:AbstractFloat`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:AbstractFloat`: Vector allocation for output from high pass
  filter.
- `h::Vector{S} where S<:AbstractFloat`: High pass filter.
- `g::Vector{S} where S<:AbstractFloat`: Low pass filter.
- `s::Bool`: Whether a shifted inverse transform should be performed.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.
"""
function isidwt_step!(v::AbstractVector{T},
                      w₁::AbstractVector{T}, w₂::AbstractVector{T},
                      h::Array{S,1}, g::Array{S,1},
                      s::Bool) where {T<:AbstractFloat, S<:AbstractFloat}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)÷2
    @assert length(h) == length(g)

    # Setup
    n = length(v)           # Parent length
    n₁ = length(w₁)         # Child length
    filtlen = length(h)     # Filter length

    # One step of inverse discrete transform
    for i in 1:n
        ℓ = mod1(i-s,n)     # Index of reconstructed vector
        j₀ = mod1(i,2)      # Pivot point to determine start index for filter
        j₁ = filtlen-j₀+1   # Index for low pass filter g
        j₂ = mod1(i+1,2)    # Index for high pass filter h
        k₁ = (i+1)>>1       # Index for approx coefs w₁
        k₂ = (i+1)>>1       # Index for detail coefs w₂
        @inbounds v[ℓ] = g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
        for j in (j₀+2):2:filtlen
            j₁ = filtlen-j+1
            j₂ = j + isodd(j) - iseven(j)
            k₁ = k₁-1 |> k₁ -> k₁≤0 ? mod1(k₁,n₁) : k₁
            k₂ = k₂+1 |> k₂ -> k₂>n₁ ? mod1(k₂,n₁) : k₂
            @inbounds v[ℓ] += g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
        end
    end
    return v
end