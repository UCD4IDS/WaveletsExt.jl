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
# ----- SDWT with allocation -----
@doc raw"""
    sdwt(x, wt[, L])

Computes the stationary discrete wavelet transform (SDWT) for `L` levels.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal,
    preferably of size 2ᴷ where ``K \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}` or `::Array{T,3}`: Output from SDWT on `x`.

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
function sdwt(x::AbstractArray{T},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert 1 ≤ ndims(x) ≤ 2
    @assert L ≤ maxtransformlevels(x) || throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    # Setup
    sz = size(x)
    N = ndims(x)+1
    k = N==2 ? L+1 : 3*L+1
    xw = Array{T,N}(undef, (sz...,k))
    # Compute transforms
    sdwt!(xw, x, wt, L)
    return xw
end

# ----- 1D SDWT without allocation -----
@doc raw"""
    sdwt!(xw, x, wt[, L])

Same as `sdwt` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: An allocated array
  of dimension `(n,L+1)` to write the outputs of `x` onto.
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Array{T,2}` or `xw::Array{T,3}`: Output from SDWT on `x`.

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
# ----- 2D SDWT with no allocation -----
function sdwt!(xw::AbstractArray{T,3},
               x::AbstractMatrix{T},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    @assert size(xw,3) == 3*L+1

    # Setup
    n, m = size(x)
    g, h = WT.makereverseqmfpair(wt, true)
    xw[:,:,end] = x
    temp = Array{T,3}(undef, (n,m,2))

    # SDWT
    for d in 0:(L-1)
        @inbounds v = xw[:,:, 3*(L-d)+1]       # Parent node
        @inbounds w₁ = @view xw[:,:, 3*(L-d)-2]          # Scaling + Scaling coefficients
        @inbounds w₂ = @view xw[:,:, 3*(L-d)-1]          # Detail + Scaling coefficients
        @inbounds w₃ = @view xw[:,:, 3*(L-d)]            # Scaling + Detail coefficients
        @inbounds w₄ = @view xw[:,:, 3*(L-d)+1]          # Detail + Detail coefficients
        @inbounds sdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
    end    
    return xw
end

# ----- ISDWT (Shift based) with allocation -----
"""
    isdwt(xw, wt[, sm])

Computes the inverse stationary discrete wavelet transform (iSDWT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: SDWT-transformed
  array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}` or `::Matrix{T}`: Inverse transformed signal.

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
function isdwt(xw::AbstractArray{T}, wt::OrthoFilter, sm::Integer) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    # Setup
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    # Transform
    isdwt!(x, xw, wt, sm)
    return x
end
# ----- ISDWT (Average based) with allocation -----
function isdwt(xw::AbstractArray{T}, wt::OrthoFilter) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    # Setup
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    # Transform
    isdwt!(x, xw, wt)
    return x
end

# ----- 1D ISDWT (Shift based) without allocation -----
"""
    isdwt!(x, xw, wt[, sm])

Same as `isdwt` but with no array allocation.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Allocation for
  reconstructed signal.
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: SDWT-transformed
  array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}` or `x::Matrix{T}`: Inverse transformed signal.

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
# ----- 2D ISDWT (Shift based) without allocation -----
function isdwt!(x::AbstractMatrix{T},
                xw::AbstractArray{T,3},
                wt::OrthoFilter,
                sm::Integer) where T<:Number
    # Sanity check
    n, m, k = size(xw)
    L = (k-1) ÷ 3
    @assert 0 ≤ log2(sm) ≤ L

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    sd = Utils.main2depthshift(sm, L)
    temp = Array{T,3}(undef, (n,m,2))
    for i in eachindex(x)
        @inbounds x[i] = xw[i]
    end

    # ISDWT
    for d in reverse(0:(L-1))
        sv = sd[d+1]
        sw = sd[d+2]
        w₁ = copy(x)
        @inbounds w₂ = @view xw[:,:,3*(L-d)-1]
        @inbounds w₃ = @view xw[:,:,3*(L-d)]
        @inbounds w₄ = @view xw[:,:,3*(L-d)+1]
        @inbounds isdwt_step!(x, w₁, w₂, w₃, w₄, d, sv, sw, h, g, temp)
    end
    return x
end

# ----- 1D ISDWT (Average based) without allocation -----
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
# ----- 2D ISDWT (Average based) without allocation -----
function isdwt!(x::AbstractMatrix{T}, xw::AbstractArray{T,3}, wt::OrthoFilter) where
                T<:Number
    # Sanity check
    n, m, k = size(xw)
    L = (k-1) ÷ 3

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    temp = Array{T,3}(undef, (n,m,2))
    for i in eachindex(x)
        @inbounds x[i] = xw[i]
    end

    # ISDWT
    for d in reverse(0:(L-1))
        w₁ = copy(x)
        @inbounds w₂ = @view xw[:,:,3*(L-d)-1]
        @inbounds w₃ = @view xw[:,:,3*(L-d)]
        @inbounds w₄ = @view xw[:,:,3*(L-d)+1]
        @inbounds isdwt_step!(x, w₁, w₂, w₃, w₄, d, h, g, temp)
    end
    return x
end

# ========== Stationary WPT ==========
# ----- SWPT with allocation -----
@doc raw"""
    swpt(x, wt[, L])

Computes `L` levels of stationary wavelet packet transform (SWPT) on `x`.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}` or `::Array{T,3}`: Output from SWPT on `x`.

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
function swpt(x::AbstractArray{T}, 
              wt::OrthoFilter, 
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert 1 ≤ ndims(x) ≤ 2
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    # Setup
    sz = size(x)
    N = ndims(x)+1
    k = N==2 ? 1<<L : 1<<(2*L)
    xw = Array{T,N}(undef, (sz..., k))
    # Transform
    swpt!(xw, x, wt, L)
    return xw
end

# ----- 1D SWPT without allocation -----
@doc raw"""
    swpt!(xw, x, wt[, L])

Same as `swpt` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: Allocation for transformed signal.
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Matrix{T}` or `xw::Array{T,3}`: Output from SWPT on `x`.

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
    @assert size(xw,2) == 1<<L
    @assert size(xw,1) == size(x,1)

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
            @inbounds v = xw[:,j₁]
            @inbounds w₁ = @view xw[:,j₁]
            @inbounds w₂ = @view xw[:,j₂]
            # Overwrite output of SDWT directly onto xw
            @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
        end
    end
    return xw
end
# ----- 2D SWPT without allocation -----
function swpt!(xw::AbstractArray{T,3},
               x::AbstractMatrix{T},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x)) where T<:Number
    @assert L ≤ maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels."))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1."))
    @assert size(xw,3) == 4^L
    @assert size(xw,1) == size(x,1)
    @assert size(xw,2) == size(x,2)

    # Setup
    n, m = size(x)
    g, h = WT.makereverseqmfpair(wt, true)
    temp = Array{T,3}(undef, (n,m,2))
    for i in eachindex(x)
        @inbounds xw[i] = x[i]
    end

    # SWPT for L levels
    for d in 0:(L-1)
        nn = 4^d            # Number of nodes at current level
        for b in 0:(nn-1)
            np = (4^L)÷nn       # Number of slices occupied by parent
            nc = np÷4           # Number of slices occupied by children
            j₁ = (4*b)*nc + 1   # Parent and scaling+scaling child index
            j₂ = (4*b+1)*nc + 1 # Scaling+detail child index
            j₃ = (4*b+2)*nc + 1 # Detail+scaling child index
            j₄ = (4*b+3)*nc + 1 # Detail+detail child index
            v = xw[:,:,j₁]
            w₁ = @view xw[:,:,j₁]
            w₂ = @view xw[:,:,j₂]
            w₃ = @view xw[:,:,j₃]
            w₄ = @view xw[:,:,j₄]
            # Overwrite output of SDWT directly onto xw
            @inbounds sdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
        end
    end
    return xw
end

# ----- ISWPT (Shift based) with allocation -----
"""
    iswpt(xw, wt[, sm])

Computes the inverse stationary wavelet packet transform (iSWPT) on `xw`.

# Arguments
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: SWPT-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}` or `::Matrix{T}`: Inverse transformed signal.

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
function iswpt(xw::AbstractArray{T}, wt::OrthoFilter, sm::Integer) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    # Setup
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    # Transform
    iswpt!(x, xw, wt, sm)
    return x
end
# ----- ISWPT (Average based) with allocation -----
function iswpt(xw::AbstractArray{T}, wt::OrthoFilter) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    # Setup
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    # Transform
    iswpt!(x, xw, wt)
    return x
end

# ----- 1D ISWPT (Shift based) without allocation -----
"""
    iswpt!(x, xw, wt[, sm])

Same as `iswpt` but with no array allocation.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Allocation for inverse
  transform.
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: SWPT-transformed
  array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}` or `x::Matrix{T}`: Inverse transformed signal.

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
# ----- 2D ISWPT (Shift based) without allocation
function iswpt!(x::AbstractMatrix{T},
                xw::AbstractArray{T,3},
                wt::OrthoFilter,
                sm::Integer) where T<:Number
    # Sanity check
    n, m, k = size(xw)
    @assert isinteger(log(4,k)) || throw(ArgumentError("Size of dimension 3 is not a power of 4."))
    @assert size(x,1) == n
    @assert size(x,2) == m
    @assert log(4,k) ≤ maxtransformlevels(x)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    xwₜ = copy(xw)
    temp = Array{T,3}(undef, (n,m,2))       # Temporary array 
    L = log(4,k) |> Int                     # Number of decomposition levels of xw
    sd = Utils.main2depthshift(sm, L)       # Shifts at each depth

    # ISWPT
    for d in reverse(0:(L-1))
        nn = 4^d
        for b in 0:(nn-1)
            sv = sd[d+1]                    # Parent shift
            sw = sd[d+2]                    # Child shift
            np = (4^L)÷nn                   # Number of slices occupied by parent
            nc = np÷4                       # Number of slices occupied by children
            j₁ = (4*b)*nc + 1               # Parent and scaling+scaling child index
            j₂ = (4*b+1)*nc + 1             # Scaling+detail child index
            j₃ = (4*b+2)*nc + 1             # Detail+scaling child index
            j₄ = (4*b+3)*nc + 1             # Detail+detail child index
            v = d==0 ? x : @view xwₜ[:,:,j₁]
            w₁ = xwₜ[:,:,j₁]
            w₂ = @view xwₜ[:,:,j₂]
            w₃ = @view xwₜ[:,:,j₃]
            w₄ = @view xwₜ[:,:,j₄]
            # Overwrite output of iSWPT directly onto temp
            @inbounds isdwt_step!(v, w₁, w₂, w₃, w₄, d, sv, sw, h, g, temp)
        end
    end
    return x
end

# ----- 1D ISWPT (Average based) withou allocation -----
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
# ----- 2D ISWPT (Average based) without allocation -----
function iswpt!(x::AbstractMatrix{T}, xw::AbstractArray{T,3}, wt::OrthoFilter) where 
                T<:Number
    # Sanity check
    n, m, k = size(xw)
    @assert isinteger(log(4,k)) || throw(ArgumentError("Size of dimension 3 is not a power of 4."))
    @assert size(x,1) == n
    @assert size(x,2) == m
    @assert log(4,k) ≤ maxtransformlevels(x)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    xwₜ = copy(xw)
    temp = Array{T,3}(undef, (n,m,2))       # Temporary array 
    L = log(4,k) |> Int                     # Number of decomposition levels of xw

    # ISWPT
    for d in reverse(0:(L-1))
        nn = 4^d
        for b in 0:(nn-1)
            np = (4^L)÷nn                   # Number of slices occupied by parent
            nc = np÷4                       # Number of slices occupied by children
            j₁ = (4*b)*nc + 1               # Parent and scaling+scaling child index
            j₂ = (4*b+1)*nc + 1             # Scaling+detail child index
            j₃ = (4*b+2)*nc + 1             # Detail+scaling child index
            j₄ = (4*b+3)*nc + 1             # Detail+detail child index
            v = d==0 ? x : @view xwₜ[:,:,j₁]
            w₁ = xwₜ[:,:,j₁]
            w₂ = @view xwₜ[:,:,j₂]
            w₃ = @view xwₜ[:,:,j₃]
            w₄ = @view xwₜ[:,:,j₄]
            # Overwrite output of iSWPT directly onto temp
            @inbounds isdwt_step!(v, w₁, w₂, w₃, w₄, d, h, g, temp)
        end
    end
    return x
end

# ========== Stationary WPD ==========
# ----- SWPD with allocation -----
@doc raw"""
    swpd(x, wt[, L])

Computes `L` levels of stationary wavelet packet decomposition (SWPD) on `x`.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal,
  preferably of size 2ᴷ where ``K \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}` or `::Array{T,3}`: Output from SWPD on `x`.

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
function swpd(x::AbstractArray{T},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert 1 ≤ ndims(x) ≤ 2
    @assert L ≤ maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    # Setup
    sz = size(x)                            # Signal size
    N = ndims(x)+1
    k = N==2 ? 1<<(L+1)-1 : sum(4 .^(0:L))  # Total number of nodes
    xw = Array{T}(undef, (sz...,k))
    # Transform
    swpd!(xw, x, wt, L)
    return xw
end

# ----- 1D SWPD without allocation -----
@doc raw"""
    swpd!(xw, x, wt[, L])

Same as `swpd` but without array allocation.

# Arguments
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: Allocation for
  transformed signal.
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Original signal,
  preferably of size 2ᴷ where ``K \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Matrix{T}` or `xw::Array{T,3}`: Output from SWPD on `x`.

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
        d = getdepth(i,:binary)
        j₁ = getchildindex(i,:left)
        j₂ = getchildindex(i,:right)
        @inbounds v = @view xw[:,i]
        @inbounds w₁ = @view xw[:,j₁]
        @inbounds w₂ = @view xw[:,j₂]
        @inbounds sdwt_step!(w₁, w₂, v, d, h, g)
    end
    return xw
end
# ----- 2D SWPD without allocation -----
function swpd!(xw::AbstractArray{T,3},
               x::AbstractMatrix{T},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    n, m, k = size(xw)
    @assert k == sum(4 .^(0:L))
    @assert n == size(x,1)
    @assert m == size(x,2)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    temp = Array{T,3}(undef, (n,m,2))
    n₁ = k - (4^L)                    # Total number of nodes excluding leaf nodes
    for i in eachindex(x)
        @inbounds xw[i] = x[i]
    end

    # SWPD
    for i in 1:n₁
        d = getdepth(i,:quad)
        j₁ = getchildindex(i,:topleft)
        j₂ = getchildindex(i,:topright)
        j₃ = getchildindex(i,:bottomleft)
        j₄ = getchildindex(i,:bottomright)
        v = @view xw[:,:,i]
        w₁ = @view xw[:,:,j₁]
        w₂ = @view xw[:,:,j₂]
        w₃ = @view xw[:,:,j₃]
        w₄ = @view xw[:,:,j₄]
        sdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
    end
end

# ----- ISWPD (Shift based by level) with allocation -----
"""
    iswpd(xw, wt, L, sm)
    iswpd(xw, wt[, L])
    iswpd(xw, wt, tree[, sm])

Computes the inverse stationary wavelet packet transform (iSWPT) on `xw`.

!!! note 
    This function might not be very useful if one is looking to reconstruct a raw
    decomposed signal. The purpose of this function would be better utilized in applications
    such as denoising, where a signal is decomposed (`swpd`) and thresholded
    (`denoise`/`denoiseall`) before being reconstructed.

# Arguments
- `xw::AbstractArray{T,2}` or `x::AbstractArray{T,3} where T<:Number`: SWPD-transformed
  array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)` or `minimum(size(xw)[1:end-1]) |>
  maxtransformlevels`) Number of levels of decomposition used for reconstruction.
- `tree::BitVector`: Binary/Quad tree for inverse transform to be computed accordingly. 
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`::Vector{T}` or `::Matrix{T}`: Inverse transformed signal.

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

**See also:** [`isdwt_step`](@ref), [`iswpt`](@ref), [`swpd`](@ref), [`iswpd!`](@ref)
"""
function iswpd(xw::AbstractArray{T}, wt::OrthoFilter, L::Integer, sm::Integer) where 
               T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    sz = size(xw)[1:(end-1)]
    return iswpd(xw, wt, maketree(sz..., L, :full), sm)
end
# ----- ISWPD (Average based by level) with allocation -----
function iswpd(xw::AbstractArray{T},
               wt::OrthoFilter,
               L::Integer = minimum(size(xw)[1:end-1]) |> maxtransformlevels) where 
               T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    sz = size(xw)[1:(end-1)]
    return iswpd(xw, wt, maketree(sz..., L, :full))
end
# ----- ISWPD (Shift based by tree) with allocation -----
function iswpd(xw::AbstractArray{T},
               wt::OrthoFilter,
               tree::BitVector,
               sm::Integer) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    sz = size(xw)[1:(end-1)]
    x = Array{T}(undef, sz)
    iswpd!(x, xw, wt, tree, sm)
    return x
end
# ----- ISWPD (Average based by tree) with allocation -----
function iswpd(xw::AbstractArray{T},
               wt::OrthoFilter,
               tree::BitVector) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    sz = size(xw)[1:(end-1)]
    x = Array{T}(undef, sz)
    iswpd!(x, xw, wt, tree)
    return x
end

# ----- ISWPD (Shift based by level) without allocation -----
"""
    iswpd!(x, xw, wt, L, sm)
    iswpd!(x, xw, wt[, L])
    iswpd!(x, xw, wt, tree, sm)
    iswpd!(x, xw, wt, tree)

Same as `iswpd` but with no array allocation.

# Arguments
- `x::AbstractVector{T}` or `x::AbstractMatrix{T} where T<:Number`: Allocated array for
  output.
- `xw::AbstractArray{T,2}` or `xw::AbstractArray{T,3} where T<:Number`: SWPD-transformed
  array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)` or `minimum(size(xw)[1:end-1]) |>
  maxtransformlevels`) Number of levels of decomposition used for reconstruction.
- `tree::BitVector`: Binary/Quad tree for inverse transform to be computed accordingly. 
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
`x::Vector{T}` or `x::Matrix{T}`: Inverse transformed signal.

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
function iswpd!(x::AbstractArray{T},
                xw::AbstractArray{T}, 
                wt::OrthoFilter, 
                L::Integer, 
                sm::Integer) where T<:Number
    # Sanity check
    @assert ndims(x) == ndims(xw)-1
    @assert size(x) == size(xw)[1:(end-1)]
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels."))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1."))
    return iswpd!(x, xw, wt, maketree(size(x)..., L, :full), sm)
end
# ----- ISWPD (Average based by level) without allocation -----
function iswpd!(x::AbstractArray{T},
                xw::AbstractArray{T}, 
                wt::OrthoFilter, 
                L::Integer = minimum(size(xw)[1:end-1]) |> maxtransformlevels) where 
                T<:Number
    # Sanity check
    @assert ndims(x) == ndims(xw)-1
    @assert size(x) == size(xw)[1:(end-1)]
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels."))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iswpd!(x, xw, wt, maketree(size(xw,1), L, :full))
end
# ----- 1D ISWPD (Shift based by tree) without allocation -----
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
    L = getdepth(m,:binary)                 # Number of decompositions from xw
    sd = Utils.main2depthshift(sm, L)       # Shifts at each depth

    # iSWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = getdepth(i,:binary)                     # Parent depth
            sv = sd[d+1]                                # Parent shift
            sw = sd[d+2]                                # Child shift
            j₁ = getchildindex(i,:left)                 # Scaling child index
            j₂ = getchildindex(i,:right)                # Detail child index
            @inbounds v = i==1 ? x : @view tmp[:,i]     # Parent node
            @inbounds w₁ = @view tmp[:,j₁]              # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]              # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
        end
    end
    return x
end
# ----- 2D ISWPD (Shift based by tree) without allocation -----
function iswpd!(x::AbstractMatrix{T},
                xw::AbstractArray{T,3},
                wt::OrthoFilter,
                tree::BitVector,
                sm::Integer) where T<:Number
    # Sanity check
    @assert size(x) == size(xw)[1:(end-1)]
    @assert isvalidtree(x, tree)

    # Setup
    n, m, k = size(xw)
    g, h = WT.makereverseqmfpair(wt, true)
    xwₜ = copy(xw)                           # Temp. array to store intermediate outputs
    temp = Array{T,3}(undef, (n,m,2))
    L = getdepth(k,:quad)                   # Number of decompositions from xw
    sd = Utils.main2depthshift(sm, L)       # Shifts at each depth

    # ISWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = getdepth(i,:quad)                       # Parent depth
            sv = sd[d+1]                                # Parent shift
            sw = sd[d+2]                                # Child shift
            j₁ = getchildindex(i,:topleft)              # Scaling child index
            j₂ = getchildindex(i,:topright)             # Detail child index
            j₃ = getchildindex(i,:bottomleft)           # Detail child index
            j₄ = getchildindex(i,:bottomright)          # Detail child index
            @inbounds v = i==1 ? x : @view xwₜ[:,:,i]    # Parent node
            @inbounds w₁ = @view xwₜ[:,:,j₁]             # Scaling child node
            @inbounds w₂ = @view xwₜ[:,:,j₂]             # Detail child node
            @inbounds w₃ = @view xwₜ[:,:,j₃]             # Detail child node
            @inbounds w₄ = @view xwₜ[:,:,j₄]             # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, w₃, w₄, d, sv, sw, h, g, temp)
        end
    end
    return x
end

# ----- 1D ISWPD (Average based by tree) without allocation -----
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
            d = getdepth(i,:binary)                     # Parent depth
            j₁ = getchildindex(i,:left)                 # Scaling child index
            j₂ = getchildindex(i,:right)                # Detail child index
            @inbounds v = i==1 ? x : @view tmp[:,i]     # Parent node
            @inbounds w₁ = @view tmp[:,j₁]              # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]              # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, d, h, g)
        end
    end
    return x
end
# ----- 2D ISWPD (Average based by tree) without allocation -----
function iswpd!(x::AbstractMatrix{T},
                xw::AbstractArray{T,3},
                wt::OrthoFilter,
                tree::BitVector) where T<:Number
    # Sanity check
    @assert size(x) == size(xw)[1:(end-1)]
    @assert isvalidtree(x, tree)

    # Setup
    n, m, k = size(xw)
    g, h = WT.makereverseqmfpair(wt, true)
    xwₜ = copy(xw)                           # Temp. array to store intermediate outputs
    temp = Array{T,3}(undef, (n,m,2))

    # ISWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = getdepth(i,:quad)                       # Parent depth
            j₁ = getchildindex(i,:topleft)              # Scaling child index
            j₂ = getchildindex(i,:topright)             # Detail child index
            j₃ = getchildindex(i,:bottomleft)           # Detail child index
            j₄ = getchildindex(i,:bottomright)          # Detail child index
            @inbounds v = i==1 ? x : @view xwₜ[:,:,i]    # Parent node
            @inbounds w₁ = @view xwₜ[:,:,j₁]             # Scaling child node
            @inbounds w₂ = @view xwₜ[:,:,j₂]             # Detail child node
            @inbounds w₃ = @view xwₜ[:,:,j₃]             # Detail child node
            @inbounds w₄ = @view xwₜ[:,:,j₄]             # Detail child node
            # Inverse transform
            @inbounds isdwt_step!(v, w₁, w₂, w₃, w₄, d, h, g, temp)
        end
    end
    return x
end

include("swt/swt_one_level.jl")
include("swt/swt_all.jl")

end # end module