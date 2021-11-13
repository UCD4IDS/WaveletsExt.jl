# ========== Single Step Autocorrelation Wavelet Transform ==========
## 1D ##
"""
    acdwt_step(v, d, h, g)

Performs one level of the autocorrelation discrete wavelet transform (acdwt) on the vector
`v`, which is the j-th level scaling coefficients (Note the 0th level scaling coefficients
is the raw signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
- `v::AbstractArray{T} where T<:Number`: Array of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::AbstractVector{T} where T<:Number` or `w₁::AbstractMatrix{T} where T<:Number`: Vector
  allocation for output from low pass filter (1D case); or matrix allocation for output from
  low + low pass filter (2D case).
- `w₂::AbstractVector{T} where T<:Number` or `w₂::AbstractMatrix{T} where T<:Number`: Vector
  allocation for output from high pass filter (1D case); or matrix allocation for output
  from low + high pass filter (2D case).
- `w₃::AbstractVector{T} where T<:Number` or `w₃::AbstractMatrix{T} where T<:Number`: Matrix
  allocation for output from high + low pass filter (2D case).
- `w₄::AbstractVector{T} where T<:Number` or `w₄::AbstractMatrix{T} where T<:Number`: Matrix
  allocation for output from high + high pass filter (2D case).

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.make_acreverseqmfpair(wt)

# One step of ACDWT
ACWT.acdwt_step(v, 0, h, g)
```

**See also:** [`acdwt`](@ref), [`iacdwt`](@ref)
"""
function acdwt_step(v::AbstractVector{T}, d::Integer, h::Array{S,1}, g::Array{S,1}) where
                   {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T,n)
    w₂ = zeros(T,n)

    acdwt_step!(w₁, w₂, v, d, h, g)
    return w₁, w₂
end

"""
    acdwt_step!(w₁, w₂, v, j, h, g)

Same with `acdwt_step` but without array allocation.

# Arguments
- `w₁::AbstractVector{T} where T<:Number` or `w₁::AbstractMatrix{T} where T<:Number`: Vector
  allocation for output from low pass filter (1D case); or matrix allocation for output from
  low + low pass filter (2D case).
- `w₂::AbstractVector{T} where T<:Number` or `w₂::AbstractMatrix{T} where T<:Number`: Vector
  allocation for output from high pass filter (1D case); or matrix allocation for output
  from low + high pass filter (2D case).
- `w₃::AbstractVector{T} where T<:Number` or `w₃::AbstractMatrix{T} where T<:Number`: Matrix
  allocation for output from high + low pass filter (2D case).
- `w₄::AbstractVector{T} where T<:Number` or `w₄::AbstractMatrix{T} where T<:Number`: Matrix
  allocation for output from high + high pass filter (2D case).
- `v::AbstractArray{T} where T<:Number`: Array of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::AbstractVector{T} where T<:Number` or `w₁::AbstractMatrix{T} where T<:Number`: Output
  from low pass filter (1D case); or output from low + low pass filter (2D case).
- `w₂::AbstractVector{T} where T<:Number` or `w₂::AbstractMatrix{T} where T<:Number`: Output
  from high pass filter (1D case); or output from low + high pass filter (2D case).
- `w₃::AbstractVector{T} where T<:Number` or `w₃::AbstractMatrix{T} where T<:Number`: Output
  from high + low pass filter (2D case).
- `w₄::AbstractVector{T} where T<:Number` or `w₄::AbstractMatrix{T} where T<:Number`: Output
  from high + high pass filter (2D case).

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
w₁ = similar(v)
w₂ = similar(v)
wt = wavelet(WT.haar)
g, h = WT.make_acreverseqmfpair(wt)

# One step of ACDWT
ACWT.acdwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`acdwt_step!`](@ref), [`acdwt`](@ref), [`iacdwt`](@ref)
"""
function acdwt_step!(w₁::AbstractVector{T},
                     w₂::AbstractVector{T},
                     v::AbstractVector{T}, 
                     d::Integer, 
                     h::Array{T,1}, 
                     g::Array{T,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)
    @assert length(h) == length(g)

    # Setup
    N = length(v)
    L = length(h)
  
    # One step of autocorrelation transform
    for i in 1:N
        t = i+(1<<d) |> t -> t>N ? mod1(t,N) : t
        i = mod1(i + (L÷2+1) * 2^d,N) # Need to shift by half the filter size because of periodicity assumption 
        @inbounds w₁[i] = g[1] * v[t]
        @inbounds w₂[i] = h[1] * v[t]
        for n in 2:L
            t = t+(1<<d) |> t -> t>N ? mod1(t,N) : t
            @inbounds w₁[i] += g[n] * v[t]
            @inbounds w₂[i] += h[n] * v[t]
        end
    end
    return w₁, w₂
end

"""
    iacdwt_step(w₁, w₂)
    iacdwt_step(w₁, w₂, w₃, w₄)

Perform one level of the inverse autocorrelation discrete wavelet transform (IACDWT) on the
vectors `w₁` and `w₂`, which are the `j+1`-th level scaling coefficients (Note that the 0th
level scaling coefficients is the raw signal).

# Arguments
- `w₁::AbstractVector{T} where T<:Number` or `w₁::AbstractMatrix{T} where T<:Number`:
  Coefficients of left child node (1D case); or coefficients from top left child node (2D
  case).
- `w₂::AbstractVector{T} where T<:Number` or `w₂::AbstractMatrix{T} where T<:Number`:
  Coefficients of right child node (1D case); or coefficients from top right child node (2D
  case).
- `w₃::AbstractVector{T} where T<:Number` or `w₃::AbstractMatrix{T} where T<:Number`:
  Coefficients from bottom left child node (2D case).
- `w₄::AbstractVector{T} where T<:Number` or `w₄::AbstractMatrix{T} where T<:Number`:
  Coefficients from bottom right child node (2D case).

# Returns
- `v::Array{T}`: Reconstructed coefficients.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.make_acreverseqmfpair(wt)

# One step of ACDWT
w₁, w₂ = ACWT.acdwt_step(v, 0, h, g)

# One step of IACDWT
v̂ = ACWT.iacdwt_step(w₁, w₂)
```

**See also:** [`iacdwt_step!`](@ref), [`acdwt_step`](@ref), [`iacdwt`](@ref)
"""
function iacdwt_step(w₁::AbstractVector{T}, w₂::AbstractVector{T}) where T<:Number
    v = similar(w₁)
    iacdwt_step!(v, w₁, w₂)
    return v
end

"""
    iacdwt_step!(v, w₁, w₂)

Same as `iacdwt_step` but without array allocation.

# Arguments
- `v::AbstractArray{T} where T<:Number`: Array allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:Number` or `w₁::AbstractMatrix{T} where T<:Number`:
  Coefficients of left child node (1D case); or coefficients from top left child node (2D
  case).
- `w₂::AbstractVector{T} where T<:Number` or `w₂::AbstractMatrix{T} where T<:Number`:
  Coefficients of right child node (1D case); or coefficients from top right child node (2D
  case).
- `w₃::AbstractVector{T} where T<:Number` or `w₃::AbstractMatrix{T} where T<:Number`:
  Coefficients from bottom left child node (2D case).
- `w₄::AbstractVector{T} where T<:Number` or `w₄::AbstractMatrix{T} where T<:Number`:
  Coefficients from bottom right child node (2D case).

# Returns
- `v::Array{T}`: Reconstructed coefficients.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.make_acreverseqmfpair(wt)

# One step of ACDWT
w₁, w₂ = ACWT.acdwt_step(v, 0, h, g)

# One step of IACDWT
v̂ = similar(v)
ACWT.iacdwt_step!(v̂, w₁, w₂)
```

**See also:** [`iacdwt_step`](@ref), [`acdwt_step`](@ref), [`iacdwt`](@ref)
"""
function iacdwt_step!(v::AbstractVector{T}, 
                      w₁::AbstractVector{T}, 
                      w₂::AbstractVector{T}) where T<:Number
    @assert length(v) == length(w₁) == length(w₂)
    for i in eachindex(v)
        @inbounds v[i] = (w₁[i]+w₂[i]) / √2
    end
end

## 2D ##
# Forward transform with allocation
function acdwt_step(v::AbstractMatrix{T}, d::Integer, h::Vector{S}, g::Vector{S}) where
                   {T<:Number, S<:Number}
    n, m = size(v)
    w₁ = Matrix{T}(undef, (n,m))
    w₂ = Matrix{T}(undef, (n,m))
    w₃ = Matrix{T}(undef, (n,m))
    w₄ = Matrix{T}(undef, (n,m))
    temp = Array{T,3}(undef, (n,m,2))
    acdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
    return w₁, w₂, w₃, w₄
end
# Forward transform without allocation
function acdwt_step!(w₁::AbstractMatrix{T}, w₂::AbstractMatrix{T},
                     w₃::AbstractMatrix{T}, w₄::AbstractMatrix{T},
                     v::AbstractMatrix{T},
                     d::Integer,
                     h::Vector{S}, g::Vector{S},
                     temp::AbstractArray{T,3}) where {T<:Number, S<:Number}
    # Sanity check
    @assert size(v) == size(w₁) == size(w₂) == size(w₃) == size(w₄)
    @assert ndims(temp) == 3
    @assert size(temp,1) == size(v,1)
    @assert size(temp,2) == size(v,2)

    # Setup
    n, m = size(w₁)

    # --- Transform ---
    # Compute acdwt for all columns
    for j in 1:m
        @inbounds temp₁ⱼ = @view temp[:,j,1]
        @inbounds temp₂ⱼ = @view temp[:,j,2]
        @inbounds vⱼ = @view v[:,j]
        @inbounds acdwt_step!(temp₁ⱼ, temp₂ⱼ, vⱼ, d, h, g)
    end
    # Compute acdwt for all rows
    for i in 1:n
        @inbounds temp₁ᵢ = @view temp[i,:,1]
        @inbounds w₁ᵢ = @view w₁[i,:]
        @inbounds w₂ᵢ = @view w₂[i,:]
        @inbounds acdwt_step!(w₁ᵢ, w₂ᵢ, temp₁ᵢ, d, h, g)
        
        @inbounds temp₂ᵢ = @view temp[i,:,2]
        @inbounds w₃ᵢ = @view w₃[i,:]
        @inbounds w₄ᵢ = @view w₄[i,:]
        @inbounds acdwt_step!(w₃ᵢ, w₄ᵢ, temp₂ᵢ, d, h, g)
    end
    return w₁, w₂, w₃, w₄
end

# Inverse transform with allocation
function iacdwt_step(w₁::AbstractMatrix{T}, w₂::AbstractMatrix{T},
                     w₃::AbstractMatrix{T}, w₄::AbstractMatrix{T}) where T<:Number
    n, m = size(w₁)
    v = Matrix{T}(undef, (n,m))
    temp = Array{T,3}(undef, (n,m,2))
    iacdwt_step!(v, w₁, w₂, w₃, w₄, temp)
    return v
end
# Inverse transform without allocation
function iacdwt_step!(v::AbstractMatrix{T},
                      w₁::AbstractMatrix{T}, w₂::AbstractMatrix{T},
                      w₃::AbstractMatrix{T}, w₄::AbstractMatrix{T},
                      temp::AbstractArray{T,3}) where T<:Number
    # Sanity check
    @assert size(v) == size(w₁) == size(w₂) == size(w₃) == size(w₄)
    @assert ndims(temp) == 3
    @assert size(temp,1) == size(v,1)
    @assert size(temp,2) == size(v,2)

    # Setup
    n, m = size(w₁)

    # --- Transform ---
    # Compute iacdwt for all rows
    for i in 1:n
        @inbounds temp₁ᵢ = @view temp[i,:,1]
        @inbounds w₁ᵢ = @view w₁[i,:]
        @inbounds w₂ᵢ = @view w₂[i,:]
        @inbounds iacdwt_step!(temp₁ᵢ, w₁ᵢ, w₂ᵢ)
        
        @inbounds temp₂ᵢ = @view temp[i,:,2]
        @inbounds w₃ᵢ = @view w₃[i,:]
        @inbounds w₄ᵢ = @view w₄[i,:]
        @inbounds iacdwt_step!(temp₂ᵢ, w₃ᵢ, w₄ᵢ)
    end
    # Compute iacdwt for all columns
    for j in 1:m
        @inbounds temp₁ⱼ = @view temp[:,j,1]
        @inbounds temp₂ⱼ = @view temp[:,j,2]
        @inbounds vⱼ = @view v[:,j]
        @inbounds iacdwt_step!(vⱼ, temp₁ⱼ, temp₂ⱼ)
    end
    return v
end