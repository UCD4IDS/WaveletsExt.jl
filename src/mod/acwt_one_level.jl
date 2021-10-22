# ========== Single Step Autocorrelation Wavelet Transform ==========
## 1D ##
"""
    acdwt_step(v, j, h, g)

Performs one level of the autocorrelation discrete wavelet transform (acdwt) on the vector
`v`, which is the j-th level scaling coefficients (Note the 0th level scaling coefficients
is the raw signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `j::Integer`: Depth level of `v`.
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
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
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
                     j::Integer, 
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
        t = i+(1<<j) |> t -> t>N ? mod1(t,N) : t
        i = mod1(i + (L÷2+1) * 2^j,N) # Need to shift by half the filter size because of periodicity assumption 
        @inbounds w₁[i] = g[1] * v[t]
        @inbounds w₂[i] = h[1] * v[t]
        for n in 2:L
            t = t+(1<<j) |> t -> t>N ? mod1(t,N) : t
            @inbounds w₁[i] += g[n] * v[t]
            @inbounds w₂[i] += h[n] * v[t]
        end
    end
    return w₁, w₂
end

"""
    iacdwt_step(w₁, w₂)

Perform one level of the inverse autocorrelation discrete wavelet transform (IACDWT) on the
vectors `w₁` and `w₂`, which are the `j+1`-th level scaling coefficients (Note that the 0th
level scaling coefficients is the raw signal).

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.

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
- `v::AbstractVector{T} where T<:Number`: Vector allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.

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
