# ========== Perform 1 step of discrete wavelet transform ==========
# ----- 1 step of dwt for 1D signals -----
# TODO: Documentation
# ! g,h is built from WT.makereverseqmfpair(wt,true)
function dwt_step!(w₁::AbstractVector{T},
                   w₂::AbstractVector{T},
                   v::AbstractVector{T},
                   h::Array{S,1},
                   g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)÷2
    @assert length(h) == length(g)

    # Setup
    n = length(v)           # Parent length
    n₁ = length(w₁)         # Child length
    filtlen = length(h)     # Filter length

    # One step of discrete transform
    for i in 1:n₁
        k₁ = 2*i-1          # Start index for low pass filtering
        k₂ = 2*i            # Start index for high pass filtering
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

# TODO: Documentation
# ! g,h is built from WT.makereverseqmfpair(wt,true)
function idwt_step!(v::AbstractVector{T},
                    w₁::AbstractVector{T},
                    w₂::AbstractVector{T},
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)÷2
    @assert length(h) == length(g)

    # Setup
    n = length(v)           # Parent length
    n₁ = length(w₁)         # Child length
    filtlen = length(h)     # Filter length

    # One step of inverse discrete transform
    for i in 1:n
        j₀ = mod1(i,2)      # Pivot point to determine start index for filter
        j₁ = filtlen-j₀+1   # Index for low pass filter g
        j₂ = mod1(i+1,2)    # Index for high pass filter h
        k₁ = (i+1)>>1       # Index for approx coefs w₁
        k₂ = (i+1)>>1       # Index for detail coefs w₂
        v[i] = g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
        for j in (j₀+2):2:filtlen
            j₁ = filtlen-j+1
            j₂ = j + isodd(j) - iseven(j)
            k₁ = k₁-1 |> k₁ -> k₁≤0 ? mod1(k₁,n₁) : k₁
            k₂ = k₂+1 |> k₂ -> k₂>n₁ ? mod1(k₂,n₁) : k₂
            v[i] += g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
        end
    end
    return v
end

# ----- 1 step of dwt for 2D signals -----
"""
    dwt_step!(y, x, filter, dcfilter, scfilter, si[; standard])

Compute 1 step of 2D discrete wavelet transform (DWT).

# Arguments
- `y::AbstactArray{T,2} where T<:Number`:
- `x::AbstactArray{T,2} where T<:Number`:
- `filter::OrthoFilter`

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to perform the standard wavelet transform.

# Returns

"""
function dwt_step!(y::AbstractArray{T,2},
                   x::AbstractArray{T,2},
                   filter::OrthoFilter,
                   dcfilter::StridedVector{S},
                   scfilter::StridedVector{S},
                   si::StridedVector{S};
                   standard::Bool = true) where {T<:Number, S<:Number}
    # Sanity check
    @assert size(x) == size(y)

    # Setup
    fw = true           # Is forward transform
    temp = similar(x)   # Temporary matrix

    # Transform
    if standard
        # Compute dwt for all rows
        @views for (tempᵢ, xᵢ) in zip(eachrow(temp), eachrow(x))
            Transforms.unsafe_dwt1level!(tempᵢ, xᵢ, filter, fw, dcfilter, scfilter, si)
        end
        # Compute dwt for all columns
        @views for (yⱼ, tempⱼ) in zip(eachcol(y), eachcol(temp))
            Transforms.unsafe_dwt1level!(yⱼ, tempⱼ, filter, fw, dcfilter, scfilter, si)
        end
    else
        # TODO: Implement non-standard transform
        error("Non-standard transform not implemented yet.")
    end
    return y
end

function idwt_step!(y::AbstractArray{T,2},
                    x::AbstractArray{T,2},
                    filter::OrthoFilter,
                    dcfilter::StridedVector{S},
                    scfilter::StridedVector{S},
                    si::StridedVector{S};
                    standard::Bool = true) where {T<:Number, S<:Number}
    # Sanity check
    @assert size(x) == size(y)

    # Setup
    fw = false          # Is inverse transform
    temp = similar(x)   # Temporary matrix

    # Transform
    if standard
        @inbounds begin
            # Compute dwt for all rows
            @views for (tempᵢ, xᵢ) in zip(eachrow(temp), eachrow(x))
                Transforms.unsafe_dwt1level!(tempᵢ, xᵢ, filter, fw, dcfilter, scfilter, si)
            end
            # Compute dwt for all columns
            @views for (yⱼ, tempⱼ) in zip(eachcol(y), eachcol(temp))
                Transforms.unsafe_dwt1level!(yⱼ, tempⱼ, filter, fw, dcfilter, scfilter, si)
            end
        end
    else
        # TODO: Implement non-standard transform
        error("Non-standard transform not implemented yet.")
    end
    return y
end

# ----- 1 step of dwt for nD signals -----
# TODO: Implement this function