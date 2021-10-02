module TIWT
export tidwt,
       tiwpt,
       tiwpd,
       itidwt,
       itiwpt,
       itiwpd

using Wavelets

using ..Utils

# ========== Single Step Translation-Invariant Wavelet Transform ==========
"""
    tidwt_step(v, d, h, g)

Performs one level of the translation-invariant discrete wavelet transform (TWDWT) on the
vector `v`, which is the `d`-th level scaling coefficients (Note the 0-th level scaling
coefficients is the raw signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
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
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of TIDWT
tidwt_step(v, 0, h, g)
```

**See also:** [`tidwt_step!`](@ref)
"""
function tidwt_step(v::AbstractVector{T}, d::Integer, h::Vector{S}, g::Vector{S}) where
                   {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T, n)
    w₂ = zeros(T, n)

    tidwt_step!(w₁, w₂, v, d, h, g)
    return w₁, w₂
end

"""
    tidwt_step!(w₁, w₂, v, d, h, g)

Same as `tidwt_step` but without array allocation.

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
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)
w₁ = zeros(8)
w₂ = zeros(8)

# One step of TIDWT
tidwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`tidwt_step`](@ref)
"""
function tidwt_step!(w₁::AbstractVector{T},
                     w₂::AbstractVector{T},
                     v::AbstractVector{T},
                     d::Integer,
                     h::Vector{S},
                     g::Vector{S}) where {T<:Number, S<:Number}
    # Variable allocations to be used in Wavelets.Transforms.filtdown! 
    n = length(v)                           # Signal length
    nc = Utils.nodelength(n, d+1)           # Node length at level d+1
    filtlen = length(h)                     # Filter length
    si = Vector{S}(undef, filtlen-1)        # Temp. filter vector

    # One step of translation-invariant transform
    for b in 0:(1<<d-1)
        # ----- Transform original block -----
        block = v[Utils.packet(d,b,n)]
        st = 2*b*nc + 1                     # Starting index in w₁, w₂
        # Decompose using low and high pass filter
        Transforms.filtdown!(g, si, w₁, st, nc, block, 1, 0, false)
        Transforms.filtdown!(h, si, w₂, st, nc, block, 1, -filtlen+1, true)
        # ----- Transform circularly shifted block -----
        block = circshift(block, 1)
        st = (2*b+1)*nc + 1                 # Starting index in w₁, w₂
        # Decompose using low and high pass filter
        Transforms.filtdown!(g, si, w₁, st, nc, block, 1, 0, false)
        Transforms.filtdown!(h, si, w₂, st, nc, block, 1, -filtlen+1, true)
    end
    return w₁, w₂
end

"""
    itidwt_step(w₁, w₂, d, g, h)

Perform one level of the inverse translation-invariant discrete wavelet transform (ITIDWT)
on the vector `w₁` and `w₂`, which are the `d+1`-th level scaling and detail coefficients
respectivaly (Note the 0-th level scaling coefficients correspond to the raw signal). The
vectors `h` and `g` are the high and low pass filters.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: `d+1`-level scaling coefficients.
- `w₂::AbstractVector{T} where T<:Number`: `d+1`-level detail coefficients.
- `d::Integer`: Depth level of output coefficients.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
`v::Vector{T}`: Output of inverse translation-invariant wavelet transform.

# Examples
```julia
using WaveletsExt

# Setup
w₁, w₂ = randn(8), randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, false)

# One step of iTIDWT
tidwt_step(w₁, w₂, 0, h, g)
```

**See also:** [`itidwt_step!`](@ref)
"""
function itidwt_step(w₁::AbstractVector{T}, 
                     w₂::AbstractVector{T}, 
                     d::Integer,
                     h::Vector{S},
                     g::Vector{S}) where {T<:Number, S<:Number}
    v = similar(w₁)
    itidwt_step!(v, w₁, w₂, d, h, g)
    return v
end

"""
    itidwt_step!(v, w₁, w₂, d, h, g)

Same as `itidwt_step` but without array allocation.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Output vector allocation.
- `w₁::AbstractVector{T} where T<:Number`: `d+1`-level scaling coefficients.
- `w₂::AbstractVector{T} where T<:Number`: `d+1`-level detail coefficients.
- `d::Integer`: Depth level of output coefficients.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
`v::AbstractVector{T}`: Output from inverse translation-invariant wavelet transform of `w₁`
and `w₂`.

# Examples
```julia
using WaveletsExt

# Setup
w₁, w₂ = randn(8), randn(8)
v = similar(w₁)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, false)

# One step of iTIDWT
tidwt_step(v, w₁, w₂, 0, h, g)
```

**See also:** [`itidwt_step`](@ref)
"""
function itidwt_step!(v::AbstractVector{T},
                      w₁::AbstractVector{T},
                      w₂::AbstractVector{T},
                      d::Integer,
                      h::Vector{S},
                      g::Vector{S}) where {T<:Number, S<:Number}
    # Variable allocations to be used in Wavelets.Transforms.filtup! 
    n = length(v)                           # Signal length
    np = Utils.nodelength(n, d)             # Node length at level d
    nc = Utils.nodelength(n, d+1)           # Node length at level d+1
    filtlen = length(h)                     # Filter length
    si = Vector{S}(undef, filtlen-1)        # Temp. filter vector
    v₁, v₂ = similar(v), similar(v)         # Temp. output storages

    # One step of translation-invariant transform
    for b in 0:(1<<d-1)
        sp = b*np + 1                       # Starting index in v
        # ----- Inverse transform original block -----
        sc = 2*b*nc + 1                     # Starting index in w₁, w₂
        # Reconstruct using low and high pass filter
        Transforms.filtup!(false, g, si, v₁, sp, np, w₁, sc, -filtlen+1, false)
        Transforms.filtup!(true, h, si, v₁, sp, np, w₂, sc, 0, true)
        # ----- Inverse transform circularly shifted block -----
        sc = (2*b+1)*nc + 1                 # Starting index in w₁, w₂
        # Reconstruct using low and high pass filter
        Transforms.filtup!(false, g, si, v₂, sp, np, w₁, sc, -filtlen+1, false)
        Transforms.filtup!(true, h, si, v₂, sp, np, w₂, sc, 0, true)
        # Adjust shift
        v₂[Utils.packet(d,b,n)] = circshift(v₂[Utils.packet(d,b,n)],-1)
    end
    # Compute result from v₁ and v₂
    for i in eachindex(v)
        @inbounds v[i] = (v₁[i] + v₂[i])/2
    end
    return v
end

# ========== Translation-Invariant DWT ==========
@doc raw"""
    tidwt(x, wt[, L])

Computes the translation-invariant discrete wavelet transform (TIDWT) for `L` levels.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from TIDWT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# TIDWT
xw = tidwt(x, wt)
```

**See also:** [`itidwt`](@ref)
"""
function tidwt(x::AbstractVector{T}, 
               wt::OrthoFilter, 
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    n = length(x)
    w = Array{T,2}(undef, (n,L))
    v = copy(x)

    # TIDWT for L levels
    @inbounds begin
        for d in 0:(L-1)
            v, w[:,L-d] = tidwt_step(v, d, h, g)
        end
    end
    return [v w]
end

"""
    itidwt(xw, wt)

Computes the inverse translation-invariant discrete wavelet transform (iTIDWT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: TIDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.

# Returns
`::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# TIDWT
xw = tidwt(x, wt)

# iTIDWT
x̂ = itidwt(xw, wt)
```

**See also:** [`tidwt`](@ref)
"""
function itidwt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Setup
    _, k = size(xw)
    g, h = WT.makereverseqmfpair(wt, false)
    x = xw[:,1]             # Output
    L = k-1                 # Number of decomposition levels

    # iTIDWT
    for i in 1:(k-1)
        @inbounds x = itidwt_step(x, xw[:,i+1], L-i, h, g)
    end
    return x
end

# ========== Translation-Invariant WPT ==========
@doc raw"""
    tiwpt(x, wt[, L])

Computes `L` levels of translation-invariant wavelet packet transform (TIWPT) on `x`.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from TIWPT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# TIWPT
xw = tiwpt(x, wt)
```

**See also:** [`itiwpt`](@ref), [`tidwt`](@ref)
"""
function tiwpt(x::AbstractVector{T},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    n = length(x)
    xw = Matrix{T}(undef, (n, 1<<L))
    xw[:,1] = x

    # TIWPT for L levels
    for d in 0:(L-1)
        nblock = 1<<d
        for b in 0:(nblock-1)
            # Parent index
            np = (1<<L)÷nblock
            i = b*np + 1
            # Children indices
            nc = np÷2
            j₁ = 2*b*nc + 1
            j₂ = (2*b+1)*nc + 1
            # Overwrite output of TIDWT directly onto xw
            xw[:,j₁], xw[:,j₂] = tidwt_step(xw[:,i], d, h, g)
        end
    end
    return xw
end

"""
    itiwpt(xw, wt)

Computes the inverse translation-invariant wavelet packet transform (iTIWPT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: TIDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.

# Returns
`::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# TIWPT
xw = tiwpt(x, wt)

# iTIWPT
x̂ = itiwpt(xw, wt)
```

**See also:** [`tiwpt`](@ref), [`itidwt`](@ref)
"""
function itiwpt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Sanity check
    n, m = size(xw)
    @assert isdyadic(m) || throw(ArgumentError("Number of columns of xw is not dyadic."))

    # Setup
    g, h = WT.makereverseqmfpair(wt, false)
    temp = copy(xw)                         # Temp. array to store intermediate outputs
    L = ndyadicscales(m)                    # Number of decompositions from xw

    # iTIWPT
    for d in (L-1):-1:0
        nblock = 1<<d
        for b in 0:(nblock-1)
            # Parent index
            np = (1<<L)÷nblock
            i = b*np + 1
            # Children indices
            nc = np÷2
            j₁ = 2*b*nc + 1
            j₂ = (2*b+1)*nc + 1
            # Overwrite output of iTIWPT directly onto temp
            temp[:,i] = itidwt_step(temp[:,j₁], temp[:,j₂], d, h, g)
        end
    end
    return temp[:,1]
end

# ========== Translation-Invariant WPD ==========
# TODO: TIWPD
function tiwpd()
    return
end

function itiwpd()
    return
end

end