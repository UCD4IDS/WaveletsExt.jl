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

# TODO: Function to compute one level of recomposition