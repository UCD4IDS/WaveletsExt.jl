module ACWT
export
    # Autocorrelation wavelet transform
    acdwt,
    acdwt!,
    acwpt,
    acwpt!,
    acwpd,
    acwpd!,
    # Inverse autocorrelation wavelet transform
    iacdwt,
    iacdwt!,
    iacwpt,
    iacwpt!,
    iacwpd,
    iacwpd!,
    # Transform on all signals
    acdwtall,
    acwptall,
    acwpdall,
    iacdwtall,
    iacwptall,
    iacwpdall

using ..Utils
using LinearAlgebra, Wavelets

# ========== Autocorrelation Discrete Wavelet Transform ==========
@doc raw"""
	acdwt(x, wt[, L])

Performs a discrete autocorrelation wavelet transform for a given signal `x`.
The signal can be 1D or 2D. The wavelet type `wt` determines the transform type.
Refer to Wavelet.jl for a list of available methods.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
    \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Array{T}`: Output from ACDWT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACDWT
acdwt(x, wt)
acdwt(x, wt, 4) # level 4 decomposition
```

**See also:** [`acdwt_step`](@ref), [`iacdwt`](@ref), [`acdwt!`](@ref)
"""
function acdwt(x::AbstractArray{T},
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
    acdwt!(xw, x, wt, L)
    return xw
end

# 1D ACDWT without allocation
@doc raw"""
    acdwt!(xw, x, wt, L)

Same as `acdwt` but without array allocation.

# Arguments
- `xw::AbstractArray{T}`: An allocated array of dimension `(n,L+1)` or `(n,m,3L+1)` to write
  the outputs of `x` onto.
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Array{T,2}`: Output from ACDWT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine, 7)
wt = wavelet(WT.haar)

# ACDWT
xw = Matrix{Float64}(undef, (128,5))
acdwt!(xw, x, wt, 4)
```

**See also:** [`acdwt`](@ref)
"""
function acdwt!(xw::AbstractArray{T,2},
                x::AbstractVector{T}, 
                wt::OrthoFilter, 
                L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) || throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))
  
    # Setup
    Pmf, Qmf = make_acreverseqmfpair(wt)
    xw[:,end] = x

    # ACDWT
    for d in 0:(L-1)
        @inbounds v = xw[:, L-d+1]       # Parent node
        w₁ = @view xw[:, L-d]            # Scaling coefficients
        w₂ = @view xw[:, L-d+1]          # Detail coefficients
        @inbounds acdwt_step!(w₁, w₂, v, d, Qmf, Pmf)
    end    
    return xw
end
# 2D ACDWT without allocation
function acdwt!(xw::AbstractArray{T,3},
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
    g, h = make_acreverseqmfpair(wt)
    xw[:,:,end] = x
    temp = Array{T,3}(undef, (n,m,2))

    # ACDWT
    for d in 0:(L-1)
        @inbounds v = xw[:,:, 3*(L-d)+1]       # Parent node
        @inbounds w₁ = @view xw[:,:, 3*(L-d)-2]          # Scaling + Scaling coefficients
        @inbounds w₂ = @view xw[:,:, 3*(L-d)-1]          # Detail + Scaling coefficients
        @inbounds w₃ = @view xw[:,:, 3*(L-d)]            # Scaling + Detail coefficients
        @inbounds w₄ = @view xw[:,:, 3*(L-d)+1]          # Detail + Detail coefficients
        @inbounds acdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
    end    
    return xw
end

## 2D ##
"""
    hacdwt(x, wt[, L=maxtransformlevels(x,2)])

Computes the column-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`vacdwt`](@ref)
"""
# function hacdwt(x::AbstractArray{T,2}, 
#                wt::OrthoFilter, 
#                L::Integer=maxtransformlevels(x,2)) where T<:Number
#     nrow, ncol = size(x)
#     W = Array{T,3}(undef,nrow,L+1,ncol)
#     for i in 1:ncol
#         @inbounds W[:,:,i] = acdwt(x[:,i],wt,L)
#     end
#     return W
# end

"""
    vacdwt(x, wt[, L=maxtransformlevels(x)])

Computes the row-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`hacdwt`](@ref)
"""
# function vacdwt(x::AbstractArray{T,2}, 
#                wt::OrthoFilter, 
#                L::Integer=maxtransformlevels(x,1)) where T<:Number
#     nrow, ncol = size(x)
#     W = Array{T,3}(undef,ncol,L+1,nrow)
#     for i in 1:nrow
#         W[:,:,i] = acdwt(x[i,:],wt,L)
#     end
#     return W
# end

# function acdwt(x::AbstractArray{T,2}, wt::OrthoFilter,
#               Lrow::Integer=maxtransformlevels(x,1),
#               Lcol::Integer=maxtransformlevels(x,2)) where T<:Number
#     nrow, ncol = size(x)
#     W3d = hacdwt(x,wt,Lcol)
#     W4d = Array{T,4}(undef,Lcol+1,ncol,Lrow+1,nrow)
#     for i in 1:Lcol+1
#         @inbounds W4d[i,:,:,:] = vacdwt(W3d[:,i,:],wt,Lrow)
#     end
#     W4d = permutedims(W4d, [4,2,3,1])
#     return W4d
# end

"""
    iacdwt(xw[, wt])

Performs the inverse autocorrelation discrete wavelet transform. Can be used for both the 1D
and 2D case.

!!! note 
    The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `dwt` and `sdwt`, but is ignored during the reconstruction of signals.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: ACDWT-transformed array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACDWT
xw = acdwt(x, wt)

# IACDWT
x̃ = iacdwt(xw)
```

**See also:** [`acdwt`](@ref), [`iacdwt_step!`](@ref)
"""
function iacdwt(xw::AbstractArray{T}, wt::Union{OrthoFilter,Nothing} = nothing) where
                T<:Number
    @assert ndims(xw) == 2 || ndims(xw) == 3
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    iacdwt!(x, xw)
    return x
end

"""
    iacdwt!(x, xw[, wt])

Similar to `iacdwt` but without array allocation.

# Arguments
- `x::AbstractArray{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`:
  Allocation for reconstructed signal.
- `xw::AbstractArray{T} where T<:Number` or `xw::AbstractArray{T,4}`: ACDWT-transformed
  array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`x::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACDWT
xw = acdwt(x, wt)

# IACDWT
x̃ = similar(x)
iacdwt!(x̃, xw)
```

**See also:** [`iacdwt`](@ref), [`acdwt`](@ref)
"""
function iacdwt!(x::AbstractVector{T}, 
                 xw::AbstractArray{T,2}, 
                 wt::Union{OrthoFilter,Nothing} = nothing) where T<:Number
    # Setup
    _, k = size(xw)
    L = k-1
    for i in eachindex(x)
        @inbounds x[i] = xw[i,1]
    end

    # IACDWT
    for d in reverse(0:(L-1))
        w₁ = copy(x)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds iacdwt_step!(x, w₁, w₂)
    end
    return x
end
# 2D IACDWT
function iacdwt!(x::AbstractMatrix{T}, 
                 xw::AbstractArray{T,3}, 
                 wt::Union{OrthoFilter, Nothing} = nothing) where
                 T<:Number
    # Sanity check
    n, m, k = size(xw)
    L = (k-1) ÷ 3

    # Setup
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
        @inbounds iacdwt_step!(x, w₁, w₂, w₃, w₄, temp)
    end
    return x
end
# function iacdwt!(x::AbstractArray{T,2},
#                  xw::AbstractArray{T,4}, 
#                  wt::Union{OrthoFilter,Nothing} = nothing) where T<:Number
#     @assert size(x,1) == size(xw,1)
#     @assert size(x,2) == size(xw,2)

#     nrow, ncol, _, Lcol = size(xw)
#     W4d = permutedims(xw,[4,2,3,1])
#     W3d = Array{T,3}(undef, nrow, Lcol, ncol)
#     for i in 1:Lcol
#         for j in 1:nrow
#             @inbounds W3d[j,i,:] = iacdwt(W4d[i,:,:,j])
#         end
#     end
#     @views for (i, xᵢ) in enumerate(eachcol(x))
#         @inbounds iacdwt!(xᵢ, W3d[:,:,i])
#     end
#     return x
# end

# ========== Autocorrelation Wavelet Packet Transform ==========
@doc raw"""
    acwpt(x, wt[, L])

Computes `L` levels of autocorrelation wavelet packet transforms (ACWPT) on `x`.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ or (2ᴷ,2ᴹ)
  where ``K, M \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Array{T}`: Output from ACWPT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPT
xw = acwpt(x, wt)
```

**See also:** [`iacwpt`](@ref), [`acdwt`](@ref), [`acwpd`](@ref)
"""
function acwpt(x::AbstractArray{T},
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
    acwpt!(xw, x, wt, L)
    return xw
end

@doc raw"""
    acwpt!(xw, x, wt[, L])

Same as `acwpt` but without array allocation.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: Allocation for transformed signal.
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ or (2ᴷ,2ᴹ)
  where ``K, M \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Array{T}`: Output from ACWPT on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPT
xw = Array{Float64,2}(undef, (128,128))
acwpt!(xw, x, wt)
```

**See also:** [`acwpt`](@ref), [`iacwpt`](@ref)
"""
function acwpt!(xw::AbstractArray{T,2},
                x::AbstractVector{T},
                wt::OrthoFilter,
                L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    @assert size(x,1) == size(xw,1)
    @assert size(xw,2) == 1<<L

    # Setup
    Pmf, Qmf = make_acreverseqmfpair(wt)
    for i in eachindex(x)
        @inbounds xw[i,1] = x[i]
    end

    # ACWPT for L levels
    for d in 0:(L-1)
        nn = 1<<d
        for b in 0:(nn-1)
            np = (1<<L)÷nn
            nc = np÷2
            j₁ = (2*b)*nc + 1   # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1 # Detail child index
            v = xw[:,j₁]
            w₁ = @view xw[:,j₁]
            w₂ = @view xw[:,j₂]
            # Overwrite output of ACDWT directly onto xw
            @inbounds acdwt_step!(w₁, w₂, v, d, Qmf, Pmf)
        end
    end
    return xw
end
# 2D ACWPT
function acwpt!(xw::AbstractArray{T,3},
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
    g, h = make_acreverseqmfpair(wt)
    temp = Array{T,3}(undef, (n,m,2))
    for i in eachindex(x)
        @inbounds xw[i] = x[i]
    end

    # ACWPT for L levels
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
            @inbounds acdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
        end
    end
    return xw
end

"""
    iacwpt(xw[, wt])

Computes the inverse autocorrelation wavelet packet transform (IACWPT) on `xw`.

!!! note 
    The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `wpt` and `swpt`, but is ignored during the reconstruction of signals.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: ACWPT-transformed array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPT
xw = acwpt(x, wt)

# IACWPT
x̃ = iacwpt(xw)
```

**See also:** [`iacdwt`](@ref), [`acwpt`](@ref)
"""
function iacwpt(xw::AbstractArray{T}, wt::Union{OrthoFilter, Nothing} = nothing) where
                T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    # Setup
    sz = size(xw)[1:(end-1)]
    N = ndims(xw)-1
    x = Array{T,N}(undef, sz)
    # Transform
    iacwpt!(x, xw)
    return x
end

# 1D IACWPT without allocation
"""
    iacwpt!(x, xw[, wt])

Same as `iacwpt` but without array allocation.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Allocated array for output.
- `xw::AbstractArray{T} where T<:Number`: ACWPD-transformed array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`x::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPT
xw = acwpt(x, wt)

# IACWPT
x̂ = similar(x)
iacwpt!(x̂, xw)
```

**See also:** [`iacwpt`](@ref)
"""
function iacwpt!(x::AbstractVector{T},
                 xw::AbstractArray{T,2}, 
                 wt::Union{OrthoFilter,Nothing} = nothing) where T<:Number
    # Sanity check
    n, m = size(xw)
    @assert isdyadic(m) || throw(ArgumentError("Number of columns of xw is not dyadic."))
    @assert ndyadicscales(m) ≤ maxtransformlevels(n) || 
            throw(ArgumentError("Number of nodes in `xw` is more than possible number of nodes at any depth for signal of length `n`"))
            
    # Setup
    temp = copy(xw)                         # Temp. array to store intermediate outputs
    L = ndyadicscales(m)                    # Number of decompositions from xw

    # IACWPT
    for d in reverse(0:(L-1))
        nn = 1<<d
        for b in 0:(nn-1)
            np = (1<<L)÷nn
            nc = np÷2
            j₁ = (2*b)*nc + 1                           # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1                         # Detail child index
            @inbounds v = d==0 ? x : @view temp[:,j₁]   # Parent node
            @inbounds w₁ = temp[:,j₁]                   # Scaling node
            @inbounds w₂ = @view temp[:,j₂]             # Detail node
            # Overwrite output of iSWPT directly onto temp
            @inbounds iacdwt_step!(v, w₁, w₂)
        end
    end
    return x
end
# 2D IACWPT
function iacwpt!(x::AbstractMatrix{T}, 
                 xw::AbstractArray{T,3}, 
                 wt::Union{OrthoFilter,Nothing} = nothing) where 
                 T<:Number
    # Sanity check
    n, m, k = size(xw)
    @assert isinteger(log(4,k)) || throw(ArgumentError("Size of dimension 3 is not a power of 4."))
    @assert size(x,1) == n
    @assert size(x,2) == m
    @assert log(4,k) ≤ maxtransformlevels(x)

    # Setup
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
            @inbounds iacdwt_step!(v, w₁, w₂, w₃, w₄, temp)
        end
    end
    return x
end

# ========== Autocorrelation Wavelet Packet Decomposition ==========
@doc raw"""
    acwpd(x, wt[, L])

Performs a discrete autocorrelation wavelet packet transform for a given signal `x`. The
wavelet type `wt` determines the transform type. Refer to Wavelet.jl for a list of available
methods.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ or (2ᴷ,2ᴹ)
  where ``K, M \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Array{T}`: Output from ACWPD on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPD
acwpd(x, wt)

acwpd(x, wt, 4)
```

**See also:** [`iacwpd`](@ref), [`acdwt`](@ref), [`acdwt_step`](@ref)
"""
function acwpd(x::AbstractArray{T},
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
    acwpd!(xw, x, wt, L)
    return xw
end

# 1D ACWPD
@doc raw"""
    acwpd!(xw, x, wt[, L])

Same as `acwpd` but without array allocation.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: Allocated array for output.
- `x::AbstractArray{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`xw::Array{T}`: Output from ACWPD on `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPD
xw = Matrix{Float64}(undef, (128, 255))
acwpd!(xw, x, wt)
acwpd!(xw, x, wt, 7)
```

**See also:** [`acwpd!`](@ref)
"""
function acwpd!(xw::AbstractArray{T,2},
                x::AbstractVector{T}, 
                wt::OrthoFilter, 
                L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    # Setup
    Pmf, Qmf = make_acreverseqmfpair(wt)
    n₀ = 1<<(L+1)-1                     # Total number of nodes
    n₁ = n₀ - (1<<L)                    # Total number of nodes excluding leaf nodes
    xw[:,1] = x

    # ACWPD
    for i in 1:n₁
        d = floor(Int, log2(i))
        j₁ = getchildindex(i,:left)
        j₂ = getchildindex(i,:right)
        @inbounds v = @view xw[:,i]
        @inbounds w₁ = @view xw[:,j₁]
        @inbounds w₂ = @view xw[:,j₂]
        @inbounds acdwt_step!(w₁, w₂, v, d, Qmf, Pmf)
    end
    return xw
end
# 2D ACWPD
function acwpd!(xw::AbstractArray{T,3},
                x::AbstractMatrix{T},
                wt::OrthoFilter,
                L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    n, m, k = size(xw)
    @assert k == sum(4 .^(0:L))
    @assert n == size(x,1)
    @assert m == size(x,2)

    # Setup
    g, h = make_acreverseqmfpair(wt)
    temp = Array{T,3}(undef, (n,m,2))
    n₁ = k - (4^L)                    # Total number of nodes excluding leaf nodes
    for i in eachindex(x)
        @inbounds xw[i] = x[i]
    end

    # ACWPD
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
        acdwt_step!(w₁, w₂, w₃, w₄, v, d, h, g, temp)
    end
end

"""
    iacwpd(xw, L)
    iacwpd(xw[, wt, L])
    iacwpd(xw, tree)
    iacwpd(xw, wt, tree)

Performs the inverse autocorrelation discrete wavelet packet transform, with respect to a
decomposition tree.

!!! note 
    The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `wpt` and `swpt`, but is ignored during the reconstruction of signals.

!!! note 
    This function might not be very useful if one is looking to reconstruct a raw
    decomposed signal. The purpose of this function would be better utilized in applications
    such as denoising, where a signal is decomposed (`swpd`) and thresholded
    (`denoise`/`denoiseall`) before being reconstructed.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: ACWPD-transformed array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition used
  for reconstruction.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly. 

# Returns
`::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPD
xw = acwpd(x, wt)

# IACWPD
x̂ = iacwpd(xw, 4)
x̂ = iacwpd(xw, wt, 4)
x̂ = iacwpd(xw, maketree(x))
x̂ = iacwpd(xw, wt, maketree(x))
```

**See also:** [`acwpd`](@ref)
"""
function iacwpd(xw::AbstractArray{T},
                wt::Union{OrthoFilter, Nothing} = nothing,
                L::Integer = maxtransformlevels(xw,1)) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    return iacwpd(xw, L)
end

function iacwpd(xw::AbstractArray{T}, L::Integer) where T<:Number
    # Sanity check
    @assert 2 ≤ ndims(xw) ≤ 3
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    sz = size(xw)[1:(end-1)]
    return iacwpd(xw, maketree(sz..., L, :full))
end

function iacwpd(xw::AbstractArray{T},
                wt::Union{OrthoFilter, Nothing},
                tree::BitVector) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    return iacwpd(xw, tree)
end

function iacwpd(xw::AbstractArray{T}, tree::BitVector) where T<:Number
    @assert 2 ≤ ndims(xw) ≤ 3
    sz = size(xw)[1:(end-1)]
    x = Array{T}(undef, sz)
    iacwpd!(x, xw, tree)
    return x
end

"""
    iacwpd!(x, xw, L)
    iacwpd!(x, xw[, wt, L])
    iacwpd!(x, xw, tree)
    iacwpd!(x, xw, wt, tree)

Same as `iacwpd` but with no array allocation.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Allocated array for output.
- `xw::AbstractArray{T} where T<:Number`: ACWPD-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition used
  for reconstruction.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly. 

# Returns
`x::Array{T}`: Inverse transformed signal.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
x = generatesignals(:heavysine)
wt = wavelet(WT.haar)

# ACWPD
xw = acwpd(x, wt)

# IACWPD
x̂ = similar(x)
iacwpd!(x̂, xw, 4)
iacwpd!(x̂, xw, wt, 4)
iacwpd!(x̂, xw, maketree(x))
iacwpd!(x̂, xw, wt, maketree(x))
```

**See also:** [`iacwpd`](@ref)
"""
function iacwpd!(x::AbstractArray{T},
                 xw::AbstractArray{T},
                 wt::Union{OrthoFilter, Nothing} = nothing,
                 L::Integer = maxtransformlevels(x)) where T<:Number
    iacwpd!(x, xw, L)
    return x
end

function iacwpd!(x::AbstractArray{T}, xw::AbstractArray{T}, L::Integer) where
                 T<:Number
    @assert ndims(x) == ndims(xw)-1
    @assert size(x) == size(xw)[1:(end-1)]
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels."))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1."))
    iacwpd!(x, xw, maketree(size(x)..., L, :full))
    return x
end

function iacwpd!(x::AbstractArray{T}, 
                 xw::AbstractArray{T},
                 wt::Union{OrthoFilter, Nothing},
                 tree::BitVector) where T<:Number
    iacwpd!(x, xw, tree)
    return x
end
# 1D IACWPD
function iacwpd!(x::AbstractVector{T}, xw::AbstractMatrix{T}, tree::BitVector) where 
                 T<:Number
    # Sanity check
    @assert size(x,1) == size(xw,1)
    @assert isvalidtree(x, tree)

    # Setup
    tmp = copy(xw)

    # iACWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = floor(Int, log2(i))                 # Parent depth
            j₁ = getchildindex(i,:left)             # Scaling child index
            j₂ = getchildindex(i,:right)            # Detail child index
            @inbounds v = i==1 ? x : @view tmp[:,i] # Parent node
            @inbounds w₁ = @view tmp[:,j₁]          # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]          # Detail child node
            # Inverse transform
            @inbounds iacdwt_step!(v, w₁, w₂)
        end
    end
    return x
end
# 2D IACWPD
function iacwpd!(x::AbstractMatrix{T}, xw::AbstractArray{T,3}, tree::BitVector) where 
                 T<:Number
    # Sanity check
    @assert size(x) == size(xw)[1:(end-1)]
    @assert isvalidtree(x, tree)

    # Setup
    n, m, k = size(xw)
    xwₜ = copy(xw)                           # Temp. array to store intermediate outputs
    temp = Array{T,3}(undef, (n,m,2))

    # IACWPD
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
            @inbounds iacdwt_step!(v, w₁, w₂, w₃, w₄, temp)
        end
    end
    return x
end

# function iacwpt(xw::AbstractArray{<:Number,2}, tree::BitVector)
#   @assert isvalidtree(xw[:,1], tree)
#   v₀ = iacwpt(xw,tree,2)
#   v₁ = iacwpt(xw,tree,3)
#   return (v₀ + v₁) / √2
# end

include("acwt/acwt_utils.jl")
include("acwt/acwt_one_level.jl")
include("acwt/acwt_all.jl")

end # end module