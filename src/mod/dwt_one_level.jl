# ========== Perform 1 step of discrete wavelet transform ==========
# ----- 1 step of dwt for 1D signals -----
"""
    dwt_step(v, h, g)

Perform one level of the discrete wavelet transform (DWT) on the vector `v`, which is the
`d`-th level scaling coefficients (Note the 0th level scaling coefficients is the raw
signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
DWT.dwt_step(v, 0, h, g)
```

**See also:** [`dwt_step!`](@ref)
"""
function dwt_step(v::AbstractVector{T}, h::Array{S,1}, g::Array{S,1}) where 
                 {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T, n÷2)
    w₂ = zeros(T, n÷2)

    dwt_step!(w₁, w₂, v, h, g)
    return w₁, w₂
end

"""
    dwt_step!(w₂, w₂, v, h, g)

Same as `dwt_step` but without array allocation.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)
w₁ = zeros(8)
w₂ = zeros(8)

# One step of SDWT
DWT.dwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`dwt_step`](@ref)
"""
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

"""
    idwt_step(w₁, w₂, h, g)

Perform one level of the inverse discrete wavelet transform (IDWT) on the vectors `w₁` and
`w₂`, which are the scaling and detail coefficients. The vectors `h` and `g` are the detail
and scaling filters.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = DWT.dwt_step(v, h, g)

# One step of ISDWT
v̂ = DWT.idwt_step(w₁, w₂, h, g)
```

**See also:** [`idwt_step!`](@ref)
"""
function idwt_step(w₁::AbstractVector{T}, 
                   w₂::AbstractVector{T}, 
                   h::Array{S,1}, 
                   g::Array{S,1}) where {T<:Number, S<:Number}
    n = length(w₁)
    v = Vector{T}(undef, 2*n)
    idwt_step!(v, w₁, w₂, h, g)
    return v
end

"""
    idwt_step!(v, w₁, w₂, h, g)

Same as `idwt_step` but without array allocation.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.


# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
v̂ = similar(v)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = DWT.dwt_step(v, h, g)

# One step of ISDWT
DWT.idwt_step!(v̂, w₁, w₂, h, g)
```

**See also:** [`idwt_step`](@ref)
"""
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
        @inbounds v[i] = g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
        for j in (j₀+2):2:filtlen
            j₁ = filtlen-j+1
            j₂ = j + isodd(j) - iseven(j)
            k₁ = k₁-1 |> k₁ -> k₁≤0 ? mod1(k₁,n₁) : k₁
            k₂ = k₂+1 |> k₂ -> k₂>n₁ ? mod1(k₂,n₁) : k₂
            @inbounds v[i] += g[j₁] * w₁[k₁] + h[j₂] * w₂[k₂]
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