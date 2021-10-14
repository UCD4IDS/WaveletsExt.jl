module SWT

export 
    # stationary wavelet transform
    sdwt,
    sdwt!,
    swpt,
    swpt!,
    swpd,
    swpd!,
    # inverse stationary wavelet transform
    isdwt,
    isdwt!,
    iswpt,
    iswpt!,
    iswpd,
    iswpd!,
    # Transforms on all signals
    sdwtall,
    swptall,
    swpdall,
    isdwtall,
    iswptall,
    iswpdall

using Wavelets

using ..Utils

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
    @assert L ≤ maxtransformlevels(x) || throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    # Setup
    n = length(x)
    xw = Array{T,2}(undef, (n,L+1))
    # Compute transforms
    sdwt!(xw, x, wt, L)
    return xw
end

@doc raw"""
    sdwt!(xw, x, wt[, L])

Same as `sdwt` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2}`: An allocated array of dimension `(n,L+1)` to write the outputs
  of `x` onto.
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Array{T,2}`: Output from SDWT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine, 7)
wt = wavelet(WT.haar)

# SDWT
xw = Matrix{Float64}(undef, (128,5))
sdwt!(xw, x, wt, 4)
```

**See also:** [`sdwt`](@ref)
"""
function sdwt!(xw::AbstractArray{T,2},
               x::AbstractVector{T}, 
               wt::OrthoFilter, 
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    xw[:,end] = x

    # SDWT
    for d in 0:(L-1)
        @inbounds v = xw[:, L-d+1]       # Parent node
        w₁ = @view xw[:, L-d]            # Scaling coefficients
        w₂ = @view xw[:, L-d+1]          # Detail coefficients
        @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
    end    
    return xw
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
    # Setup
    n = size(xw,1)
    x = Vector{T}(undef, n)
    # Transform
    isdwt!(x, xw, wt, sm)
    return x
end

function isdwt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Setup
    n = size(xw,1)
    x = Vector{T}(undef, n)
    # Transform
    isdwt!(x, xw, wt)
    return x
end

"""
    isdwt!(x, xw, wt[, sm])

Same as `isdwt` but with no array allocation.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Allocation for reconstructed signal.
- `xw::AbstractArray{T,2} where T<:Number`: SDWT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SDWT
xw = sdwt(x, wt)

#  Shift-based iSDWT
x̂ = similar(x)
isdwt!(x̂, xw, wt, 5)

# Average-based iSDWT
isdwt(x̂, xw, wt)
```

**See also:** [`isdwt`](@ref)
"""
function isdwt!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                sm::Integer) where T<:Number
    # Sanity check
    _, k = size(xw)
    L = k-1
    @assert 0 ≤ log2(sm) < L

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    sd = Utils.main2depthshift(sm, L)
    for i in eachindex(x)
        @inbounds x[i] = xw[i,1]
    end

    # ISDWT
    for d in reverse(0:(L-1))
        sv = sd[d+1]
        sw = sd[d+2]
        w₁ = copy(x)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds isdwt_step!(x, w₁, w₂, d, sv, sw, h, g)
    end
    return x
end

# Average-based ISDWT
function isdwt!(x::AbstractVector{T}, xw::AbstractArray{T,2}, wt::OrthoFilter) where 
                T<:Number
    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    _, k = size(xw)
    L = k-1
    for i in eachindex(x)
        @inbounds x[i] = xw[i,1]
    end

    # ISDWT
    for d in reverse(0:(L-1))
        w₁ = copy(x)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds isdwt_step!(x, w₁, w₂, d, h, g)
    end
    return x
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
              wt::OrthoFilter, 
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    # Setup
    n = length(x)
    xw = Matrix{T}(undef, (n, 1<<L))
    # Transform
    swpt!(xw, x, wt, L)
    return xw
end

@doc raw"""
    swpt!(xw, x, wt[, L])

Same as `swpt` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: Allocation for transformed signal.
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Matrix{T}`: Output from SWPT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPT
xw = Array{Float64,2}(undef, (128,128))
swpt!(xw, x, wt)
```

**See also:** [`swpt`](@ref)
"""
function swpt!(xw::AbstractArray{T,2},
               x::AbstractVector{T}, 
               wt::OrthoFilter,                        
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    for i in eachindex(x)
        @inbounds xw[i,1] = x[i]
    end

    # SWPT for L levels
    for d in 0:(L-1)
        nn = 1<<d               # Number of nodes at current level
        for b in 0:(nn-1)
            np = (1<<L)÷nn      # Parent node length
            nc = np÷2           # Child node length
            j₁ = (2*b)*nc + 1   # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1 # Detail child index
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
- `xw::AbstractArray{T,2} where T<:Number`: SWPT-transformed array.
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
    # Setup
    n = size(xw,1)
    x = Vector{T}(undef, n)
    # Transform
    iswpt!(x, xw, wt, sm)
    return x
end

function iswpt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    # Setup
    n = size(xw,1)
    x = Vector{T}(undef, n)
    # Transform
    iswpt!(x, xw, wt)
    return x
end

"""
    iswpt!(x, xw, wt[, sm])

Same as `iswpt` but with no array allocation.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Allocation for inverse transform.
- `xw::AbstractArray{T,2} where T<:Number`: SWPT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPT
xw = swpt(x, wt)

# Shift-based iSWPT
x̂ = similar(x)
iswpt!(x̂, xw, wt, 5)

# Average-based iSWPT
iswpt!(x̂, xw, wt)
```

**See also:** [`iswpt`](@ref)
"""
function iswpt!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                sm::Integer) where T<:Number
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
            @inbounds v = d==0 ? x : @view temp[:,j₁]
            @inbounds w₁ = temp[:,j₁]
            @inbounds w₂ = @view temp[:,j₂]
            # Overwrite output of iSWPT directly onto temp
            @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
        end
    end
    return x
end

# Average-based ISWPT
function iswpt!(x::AbstractVector{T}, xw::AbstractArray{T,2}, wt::OrthoFilter) where 
                T<:Number
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
            np = (1<<L)÷nn                              # Parent node length
            nc = np÷2                                   # Child node length
            j₁ = (2*b)*nc + 1                           # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1                         # Detail child index
            @inbounds v = d==0 ? x : @view temp[:,j₁]   # Parent node
            @inbounds w₁ = temp[:,j₁]                   # Scaling node
            @inbounds w₂ = @view temp[:,j₂]             # Detail node
            # Overwrite output of iSWPT directly onto temp
            @inbounds isdwt_step!(v, w₁, w₂, d, h, g)
        end
    end
    return x
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
    n = length(x)                       # Signal length
    n₀ = 1<<(L+1)-1                     # Total number of nodes
    xw = Matrix{T}(undef, (n, n₀))
    # Transform
    swpd!(xw, x, wt, L)
end

@doc raw"""
    swpd!(xw, x, wt[, L])

Same as `swpd` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: Allocation for transformed signal.
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Matrix{T}`: Output from SWPD on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPD
xw = Matrix{T}(undef, (128, 255))
swpd!(xw, x, wt)
```

**See also:** [`swpd`](@ref)
"""
function swpd!(xw::AbstractArray{T,2},
               x::AbstractVector{T}, 
               wt::OrthoFilter, 
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    n₀ = 1<<(L+1)-1                     # Total number of nodes
    n₁ = n₀ - (1<<L)                    # Total number of nodes excluding leaf nodes
    for i in eachindex(x)
        @inbounds xw[i,1] = x[i]
    end

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
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly. 
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
    return iswpd(xw, wt, maketree(size(xw,1), L, :full), sm)
end

function iswpd(xw::AbstractArray{T,2},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(xw,1)) where T<:Number
    return iswpd(xw, wt, maketree(size(xw,1), L, :full))
end

function iswpd(xw::AbstractArray{T,2},
               wt::OrthoFilter,
               tree::BitVector,
               sm::Integer) where T<:Number
    n = size(xw,1)
    x = Vector{T}(undef, n)
    iswpd!(x, xw, wt, tree, sm)
    return x
end

function iswpd(xw::AbstractArray{T,2},
               wt::OrthoFilter,
               tree::BitVector) where T<:Number
    n = size(xw,1)
    x = Vector{T}(undef, n)
    iswpd!(x, xw, wt, tree)
    return x
end

"""
    iswpd!(x, xw, wt, L, sm)
    iswpd!(x, xw, wt[, L])
    iswpd!(x, xw, wt, tree, sm)
    iswpd!(x, xw, wt, tree)

Same as `iswpd` but with no array allocation.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Allocated array for output.
- `xw::AbstractArray{T,2} where T<:Number`: SWPD-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition used
  for reconstruction.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly. 
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# SWPD
xw = swpd(x, wt)

# ISWPD
x̂ = similar(x)
iswpd!(x̂, xw, wt, 4, 5)
iswpd!(x̂, xw, wt, maketree(x), 5)
iswpd!(x̂, xw, wt, 4)
iswpd!(x̂, xw, wt, maketree(x))
```

**See also:** [`iswpd`](@ref)
"""
function iswpd!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                L::Integer, 
                sm::Integer) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iswpd!(x, xw, wt, maketree(size(xw,1), L, :full), sm)
end

# Average-based ISWPD by level
function iswpd!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                L::Integer = maxtransformlevels(xw,1)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iswpd!(x, xw, wt, maketree(size(xw,1), L, :full))
end

# Shift-based iSWPD by tree
function iswpd!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                tree::BitVector, 
                sm::Integer) where T<:Number
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
            d = floor(Int, log2(i))                     # Parent depth
            sv = sd[d+1]                                # Parent shift
            sw = sd[d+2]                                # Child shift
            j₁ = left(i)                                # Scaling child index
            j₂ = right(i)                               # Detail child index
            @inbounds v = i==1 ? x : @view tmp[:,i]     # Parent node
            @inbounds w₁ = @view tmp[:,j₁]              # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]              # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
        end
    end
    return tmp[:,1]
end

# Average-based iSWPD by tree
function iswpd!(x::AbstractVector{T},
                xw::AbstractArray{T,2}, 
                wt::OrthoFilter, 
                tree::BitVector) where T<:Number
    # Sanity check
    @assert isvalidtree(xw[:,1], tree)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    tmp = copy(xw)

    # iSWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = floor(Int, log2(i))                     # Parent depth
            j₁ = left(i)                                # Scaling child index
            j₂ = right(i)                               # Detail child index
            @inbounds v = i==1 ? x : @view tmp[:,i]     # Parent node
            @inbounds w₁ = @view tmp[:,j₁]              # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]              # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, d, h, g)
        end
    end
    return tmp[:,1]
end

include("swt_one_level.jl")
include("swt_all.jl")

end # end module