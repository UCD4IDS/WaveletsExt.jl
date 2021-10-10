module SWT

export 
    # stationary wavelet transform
    sdwt,
    swpt,
    swpd,
    # inverse stationary wavelet transform
    isdwt,
    iswpt,
    iswpd

using Wavelets

using ..Utils

# ========== Single Step Stationary Wavelet Transform ==========
"""
    sdwt_step(v, d, h, g)

Perform one level of the stationary discrete wavelet transform (SDWT) on the vector `v`,
which is the `d`-th level scaling coefficients (Note the 0th level scaling coefficients is
the raw signal). The vectors `h` and `g` are the detail and scaling filters.

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

# One step of SDWT
SWT.sdwt_step(v, 0, h, g)
```

**See also:** [`sdwt_step!`](@ref)
"""
function sdwt_step(v::AbstractVector{T}, d::Integer, h::Array{S,1}, g::Array{S,1}) where 
                  {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T, n)
    w₂ = zeros(T, n)

    sdwt_step!(w₁, w₂, v, d, h, g)
    return w₁, w₂
end

"""
    sdwt_step!(w₁, w₂, v, d, h, g)

Same as `sdwt_step` but without array allocation.

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

# One step of SDWT
SWT.sdwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`sdwt_step`](@ref)
"""
function sdwt_step!(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T},
                    v::AbstractVector{T},
                    d::Integer, 
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)
    @assert length(h) == length(g)
    
    # Setup
    n = length(v)                   # Signal length
    filtlen = length(h)             # Filter length

    # One step of stationary transform
    for i in 1:n
        k₁ = mod1(i-(1<<d),n)       # Start index for low pass filtering
        k₂ = i                      # Start index for high pass filtering
        @inbounds w₁[i] = g[end] * v[k₁]
        @inbounds w₂[i] = h[1] * v[k₂]
        for j in 2:filtlen
            k₁ = k₁+(1<<d) |> k₁ -> k₁>n ? mod1(k₁,n) : k₁
            k₂ = k₂-(1<<d) |> k₂ -> k₂≤0 ? mod1(k₂,n) : k₂
            @inbounds w₁[i] += g[end-j+1] * v[k₁]
            @inbounds w₂[i] += h[j] * v[k₂]
        end
    end
    return w₁, w₂
end

"""
    isdwt_step(w₁, w₂, d, h, g)
    isdwt_step(w₁, w₂, d, sv, sw, h, g)

Perform one level of the inverse stationary discrete wavelet transform (ISDWT) on the
vectors `w₁` and `w₂`, which are the `d+1`-th level scaling coefficients (Note the 0th level
scaling coefficients is the raw signal). The vectors `h` and `g` are the detail and scaling
filters.

!!! note
    One can decide to choose the average-based inverse transform or the shift-based inverse
    transform. For shift based, one needs to specify the shifts of the parent and children
    nodes; for the average based, the output is the average of all possible shift-based
    inverse transform outputs.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
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
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
v̂ = SWT.isdwt_step(w₁, w₂, 0, h, g)         # Average based
ṽ = SWT.isdwt_step(w₁, w₂, 0, 0, 1, h, g)   # Shift based
```

**See also:** [`isdwt_step!`](@ref)
"""
function isdwt_step(w₁::AbstractVector{T},
                    w₂::AbstractVector{T},
                    d::Integer,
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Setup
    v₁ = similar(w₁)        # Allocation for isdwt_step of w₁, w₂ with no additional shift
    v₂ = similar(w₁)        # Allocation for isdwt_step of w₁, w₂ with additional shift
    nd = 1 << d             # Number of blocks for parent node

    # isdwt_step for each shift
    for sv in 0:(nd-1)
        sw₁ = sv            # Shift of w₁, w₂ with no additional shift
        sw₂ = sv + 1<<d     # Shift of w₁, w₂ with addtional shift
        isdwt_step!(v₁, w₁, w₂, d, sv, sw₁, h, g)   # Without shift
        isdwt_step!(v₂, w₁, w₂, d, sv, sw₂, h, g)   # With shift
    end
    return (v₁+v₂)/2        # Average the results of v₁, v₂
end

function isdwt_step(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T}, 
                    d::Integer, 
                    sv::Integer, 
                    sw::Integer, 
                    h::Array{S,1}, 
                    g::Array{S,1}) where {T<:Number, S<:Number}
    v = similar(w₁)
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
    return v
end

"""
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)

Same as `isdwt_step` but without array allocation.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
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
ṽ = similar(v)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
SWT.isdwt_step!(v̂, w₁, w₂, 0, h, g)          # Average based
SWT.isdwt_step!(ṽ, w₁, w₂, 0, 0, 1, h, g)    # Shift based
```

**See also:** [`isdwt_step`](@ref)
"""
function isdwt_step!(v::AbstractVector{T}, 
                     w₁::AbstractVector{T}, 
                     w₂::AbstractVector{T}, 
                     d::Integer, 
                     sv::Integer, 
                     sw::Integer,
                     h::Array{S,1}, 
                     g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert sw ≥ sv

    # Setup
    n = length(v)               # Signal length
    filtlen = length(h)         # filter length
    ip = sv + 1                 # Parent start index
    sp = 1 << d                 # Parent step size
    ic = sw + 1                 # Child start index
    sc = 1 << (d+1)             # Child step size
    
    # One step of inverse SDWT
    for (t, m) in enumerate(ip:sp:n)
        i₀ = mod1(t,2)                                  # Pivot point for filter
        i₁ = filtlen-i₀+1                               # Index for low pass filter g
        i₂ = mod1(t+1,2)                                # Index for high pass filter h
        j = sw==sv ? mod1(m-1<<d,n) : mod1(m+sp-1<<d,n) # Position of v, shift needed if sw≠sv
        k₁ = ((t-1)>>1) * sc + ic                       # Index for approx coefs w₁
        k₂ = ((t-1)>>1) * sc + ic                       # Index for detail coefs w₂
        @inbounds v[j] = g[i₁] * w₁[k₁] + h[i₂] * w₂[k₂]
        for i in (i₀+2):2:filtlen
            i₁ = filtlen-i+1
            i₂ = i + isodd(i) - iseven(i)
            k₁ = k₁-(1<<(d+1)) |> k₁ -> k₁≤0 ? mod1(k₁,n) : k₁
            k₂ = k₂+(1<<(d+1)) |> k₂ -> k₂>n ? mod1(k₂,n) : k₂
            @inbounds v[j] += g[i₁] * w₁[k₁] + h[i₂] * w₂[k₂]
        end
    end
    return v
end

# ========== Stationary DWT ==========
@doc raw"""
    sdwt(x, wt[, L])

Computes the stationary discrete wavelet transform (SDWT) for `L` levels.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
    \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from SDWT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SDWT
xw = sdwt(x, wt)
```    

**See also:** [`swpd`](@ref), [`swpt`](@ref), [`isdwt`](@ref)
"""
function sdwt(x::AbstractVector{T}, 
              wt::OrthoFilter, 
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    n = length(x)
    w = Matrix{T}(undef, (n, L+1))
    w[:,end] = x

    # SDWT
    for d in 0:(L-1)
        @inbounds v = w[:, L-d+1]       # Parent node
        w₁ = @view w[:, L-d]            # Scaling coefficients
        w₂ = @view w[:, L-d+1]          # Detail coefficients
        @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
    end    
    return w
end

# Shift-based ISDWT
"""
    isdwt(xw, wt[, sm])

Computes the inverse stationary discrete wavelet transform (iSDWT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: SDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SDWT
xw = sdwt(x, wt)

#  Shift-based iSDWT
x̂ = isdwt(xw, wt, 5)

# Average-based iSDWT
x̃ = isdwt(xw, wt)
```

**See also:** [`isdwt_step`](@ref), [`iswpt`](@ref), [`sdwt`](@ref)
"""
function isdwt(xw::AbstractArray{T,2}, wt::OrthoFilter, sm::Integer) where T<:Number
    # Sanity check
    _, k = size(xw)
    L = k-1
    @assert 0 ≤ log2(sm) < L

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    sd = Utils.main2depthshift(sm, L)
    v = xw[:, 1]

    for d in reverse(0:(L-1))
        sv = sd[d+1]
        sw = sd[d+2]
        w₁ = copy(v)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
    end
    return v
end

# Average-based ISDWT
function isdwt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    _, k = size(xw)
    v = xw[:,1]
    L = k-1

    # ISDWT
    for d in reverse(0:(L-1))
        w₁ = copy(v)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds v = isdwt_step(w₁, w₂, d, h, g)
    end
    return v
end

# ========== Stationary WPT ==========
@doc raw"""
    swpt(x, wt[, L])

Computes `L` levels of stationary wavelet packet transform (SWPT) on `x`.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from SWPT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPT
xw = swpt(x, wt)
```

**See also:** [`sdwt`](@ref), [`swpd`](@ref)
"""
function swpt(x::AbstractVector{T}, 
              wt::DiscreteWavelet,                        
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

    # SWPT for L levels
    for d in 0:(L-1)
        nn = 1<<d               # Number of nodes at current level
        for b in 0:(nn-1)
            np = (1<<L)÷nn      # Parent node length
            nc = np÷2           # Child node length
            j₁ = (2*b)*nc + 1               # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1             # Detail child index
            v = xw[:,j₁]
            w₁ = @view xw[:,j₁]
            w₂ = @view xw[:,j₂]
            # Overwrite output of SDWT directly onto xw
            @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
        end
    end
    return xw
end

# Shift-based ISWPT
"""
    iswpt(xw, wt[, sm])

Computes the inverse stationary wavelet packet transform (iSWPT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: TIDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPT
xw = swpt(x, wt)

# Shift-based iSWPT
x̂ = iswpt(xw, wt, 5)

# Average-based iSWPT
x̃ = iswpt(xw, wt)
```

**See also:** [`isdwt_step`](@ref), [`isdwt`](@ref), [`swpt`](@ref)
"""
function iswpt(xw::AbstractArray{T,2}, wt::OrthoFilter, sm::Integer) where T<:Number
    # Sanity check
    n, m = size(xw)
    @assert isdyadic(m) || throw(ArgumentError("Number of columns of xw is not dyadic."))
    @assert ndyadicscales(m) ≤ maxtransformlevels(n) || 
            throw(ArgumentError("Number of nodes in `xw` is more than possible number of nodes at any depth for signal of length `n`"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    temp = copy(xw)                         # Temp. array to store intermediate outputs
    L = ndyadicscales(m)                    # Number of decompositions from xw
    sd = Utils.main2depthshift(sm, L)       # Shifts at each depth

    # ISWPT
    for d in reverse(0:(L-1))
        nn = 1<<d
        for b in 0:(nn-1)
            sv = sd[d+1]                    # Parent shift
            sw = sd[d+2]                    # Child shift
            np = (1<<L)÷nn                  # Parent node length
            nc = np÷2                       # Child node length
            j₁ = (2*b)*nc + 1               # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1             # Detail child index
            @inbounds v = @view temp[:,j₁]
            @inbounds w₁ = temp[:,j₁]
            @inbounds w₂ = @view temp[:,j₂]
            # Overwrite output of iSWPT directly onto temp
            @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
        end
    end
    return temp[:,1]
end

# Average-based ISWPT
function iswpt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Sanity check
    n, m = size(xw)
    @assert isdyadic(m) || throw(ArgumentError("Number of columns of xw is not dyadic."))
    @assert ndyadicscales(m) ≤ maxtransformlevels(n) || 
            throw(ArgumentError("Number of nodes in `xw` is more than possible number of nodes at any depth for signal of length `n`"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    temp = copy(xw)                         # Temp. array to store intermediate outputs
    L = ndyadicscales(m)                    # Number of decompositions from xw

    # ISWPT
    for d in reverse(0:(L-1))
        nn = 1<<d
        for b in 0:(nn-1)
            np = (1<<L)÷nn                  # Parent node length
            nc = np÷2                       # Child node length
            j₁ = (2*b)*nc + 1               # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1             # Detail child index
            @inbounds w₁ = temp[:,j₁]       # Scaling node
            @inbounds w₂ = @view temp[:,j₂] # Detail node
            # Overwrite output of iSWPT directly onto temp
            @inbounds temp[:,j₁] = isdwt_step(w₁, w₂, d, h, g)
        end
    end
    return temp[:,1]
end

# ========== Stationary WPD ==========
@doc raw"""
    swpd(x, wt[, L])

Computes `L` levels of stationary wavelet packet decomposition (SWPD) on `x`.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from SWPD on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPD
xw = swpd(x, wt)
```

**See also:** [`iswpd`](@ref), [`swpt`](@ref)
"""
function swpd(x::AbstractVector{T}, 
              wt::OrthoFilter, 
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    n = length(x)                       # Size of signal
    n₀ = 1<<(L+1)-1                     # Total number of nodes
    n₁ = n₀ - (1<<L)                    # Total number of nodes excluding leaf nodes
    xw = Matrix{T}(undef, (n, n₀))      # Output allocation
    xw[:,1] = x

    # SWPD
    for i in 1:n₁
        d = floor(Int, log2(i))
        j₁ = left(i)
        j₂ = right(i)
        @inbounds v = @view xw[:,i]
        @inbounds w₁ = @view xw[:,j₁]
        @inbounds w₂ = @view xw[:,j₂]
        @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
    end
    return xw
end

# Shift-based iSWPD by level
"""
    iswpd(xw, wt, L, sm)
    iswpd(xw, wt[, L])
    iswpd(xw, wt, tree[, sm])

Computes the inverse stationary wavelet packet transform (iSWPT) on `xw`.

!!! note
    This function might not be very useful if one is looking to reconstruct a raw decomposed
    signal. The purpose of this function would be better utilized in applications such as
    denoising, where a signal is decomposed (`swpd`) and thresholded
    (`denoise`/`denoiseall`) before being reconstructed.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: TIDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition used
  for reconstruction.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPD
xw = swpt(x, wt)

# Shift-based iSWPD
x̂ = iswpd(xw, wt, maxtransformlevels(xw,1), 5)

# Average-based iSWPD
x̃ = iswpd(xw, wt)
```

**See also:** [`isdwt_step`](@ref), [`iswpt`](@ref), [`swpd`](@ref)
"""
function iswpd(xw::AbstractArray{T,2}, wt::OrthoFilter, L::Integer, sm::Integer) where 
               T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iswpd(xw, wt, maketree(size(xw,1), L, :full), sm)
end

# Average-based ISWPD by level
function iswpd(xw::AbstractArray{T,2}, 
               wt::OrthoFilter, 
               L::Integer = maxtransformlevels(xw,1)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iswpd(xw, wt, maketree(size(xw,1), L, :full))
end

# Shift-based iSWPD by tree
function iswpd(xw::AbstractArray{T,2}, wt::OrthoFilter, tree::BitVector, sm::Integer) where
               T<:Number
    # Sanity check
    @assert isvalidtree(xw[:,1], tree)

    # Setup
    _, m = size(xw)
    g, h = WT.makereverseqmfpair(wt, true)
    tmp = copy(xw)                          # Temp. array to store intermediate outputs
    L = floor(Int, log2(m))                 # Number of decompositions from xw
    sd = Utils.main2depthshift(sm, L)       # Shifts at each depth

    # iSWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = floor(Int, log2(i))         # Parent depth
            sv = sd[d+1]                    # Parent shift
            sw = sd[d+2]                    # Child shift
            j₁ = left(i)                    # Scaling child index
            j₂ = right(i)                   # Detail child index
            @inbounds v = @view tmp[:,i]    # Parent node
            @inbounds w₁ = @view tmp[:,j₁]  # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]  # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
        end
    end
    return tmp[:,1]
end

# Average-based iSWPD by tree
function iswpd(xw::AbstractArray{T,2}, wt::OrthoFilter, tree::BitVector) where T<:Number
    # Sanity check
    @assert isvalidtree(xw[:,1], tree)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    tmp = copy(xw)

    # iSWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = floor(Int, log2(i))         # Parent depth
            j₁ = left(i)                    # Scaling child index
            j₂ = right(i)                   # Detail child index
            @inbounds w₁ = @view tmp[:,j₁]  # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]  # Detail child node
            # Inverse transform
            @inbounds tmp[:,i] = isdwt_step(w₁, w₂, d, h, g)
        end
    end
    return tmp[:,1]
end

end # end module