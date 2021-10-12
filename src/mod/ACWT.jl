module ACWT
export
    # Autocorrelation wavelet transform
    acdwt, 
    acwpt,
    acwpd,
    # Inverse autocorrelation wavelet transform
    iacdwt,
    iacwpt,
    iacwpd

using ..Utils
using LinearAlgebra, Wavelets

# ========== ACWT Utilities ==========
"""
    autocorr(f::OrthoFilter)

Generates the autocorrelation filter for a given wavelet filter.
"""
function autocorr(f::OrthoFilter)
    H = WT.qmf(f)
    l = length(H)
    result = zeros(l - 1)
    for k in 1:(l - 1)
        for i in 1:(l - k)
            @inbounds result[k] += H[i] * H[i + k]
        end
        result[k] *= 2
    end
    return result
end

"""
    pfilter(f::OrthoFilter)

Generates the high-pass autocorrelation filter

**See also:** [`qfilter`](@ref), [`autocorr`](@ref)
"""
function pfilter(f::OrthoFilter)
    a = autocorr(f)
    c1 = 1 / sqrt(2)
    c2 = c1 / 2
    b = c2 * a
    return vcat(reverse(b), c1, b)
end

"""
    qfilter(f::OrthoFilter)

Generates the low-pass autocorrelation filter.

**See also:** [`pfilter`](@ref), [`autocorr`](@ref)
"""
function qfilter(f::OrthoFilter)
    a = autocorr(f)
    c1 = 1 / sqrt(2)
    c2 = c1 / 2
    b = -c2 * a
    return vcat(reverse(b), c1, b)
end

"""
    make_acqmfpair(f::OrthoFilter)

Generates the autocorrelation quadratic mirror filters.

**See also:** [`make_acreverseqmfpair`](@ref), [`pfilter`](@ref), [`qfilter`](@ref)
"""
function make_acqmfpair(f::OrthoFilter)
    pmfilter, qmfilter = pfilter(f), qfilter(f)
    return pmfilter, qmfilter
end

"""
    make_acreverseqmfpair(f::OrthoFilter)

Generates the reverse autocorrelation quadratic mirror filters.

**See also:** [`make_acqmfpair`](@ref), [`pfilter`](@ref), [`qfilter`](@ref)
"""
function make_acreverseqmfpair(f::OrthoFilter)
    pmf, qmf = make_acqmfpair(f)
    return reverse(pmf), reverse(qmf)
end

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

# ========== Autocorrelation Discrete Wavelet Transform ==========
@doc raw"""
	acdwt(x, wt[, L])
    acdwt(x, wt[, Lrow=maxtransformlevels(x[1,:]), Lcol=maxtransformlevels(x[:,1])])

Performs a discrete autocorrelation wavelet transform for a given signal `x`.
The signal can be 1D or 2D. The wavelet type `wt` determines the transform type.
Refer to Wavelet.jl for a list of available methods.

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

# ACDWT
acdwt(x, wt)
acdwt(x, wt, 4) # level 4 decomposition
```

**See also:** [`acdwt_step`](@ref), [`iacdwt`](@ref)
"""
function acdwt(x::AbstractVector{T}, 
               wt::OrthoFilter, 
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) || throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))
  
    # Setup
    n = length(x)
    Pmf, Qmf = make_acreverseqmfpair(wt)
    w = zeros(T, (n, L+1))
    w[:,end] = x

    # ACDWT
    for d in 0:(L-1)
        @inbounds v = w[:, L-d+1]       # Parent node
        w₁ = @view w[:, L-d]            # Scaling coefficients
        w₂ = @view w[:, L-d+1]          # Detail coefficients
        @inbounds acdwt_step!(w₁, w₂, v, d, Qmf, Pmf)
    end    
    return w
end

## 2D ##
"""
    hacdwt(x, wt[, L=maxtransformlevels(x,2)])

Computes the column-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`vacdwt`](@ref)
"""
function hacdwt(x::AbstractArray{T,2}, 
               wt::OrthoFilter, 
               L::Integer=maxtransformlevels(x,2)) where T<:Number
    nrow, ncol = size(x)
    W = Array{T,3}(undef,nrow,L+1,ncol)
    for i in 1:ncol
        @inbounds W[:,:,i] = acdwt(x[:,i],wt,L)
    end
    return W
end

"""
    vacdwt(x, wt[, L=maxtransformlevels(x)])

Computes the row-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`hacdwt`](@ref)
"""
function vacdwt(x::AbstractArray{T,2}, 
               wt::OrthoFilter, 
               L::Integer=maxtransformlevels(x,1)) where T<:Number
    nrow, ncol = size(x)
    W = Array{T,3}(undef,ncol,L+1,nrow)
    for i in 1:nrow
        W[:,:,i] = acdwt(x[i,:],wt,L)
    end
    return W
end

function acdwt(x::AbstractArray{T,2}, wt::OrthoFilter,
              Lrow::Integer=maxtransformlevels(x,1),
              Lcol::Integer=maxtransformlevels(x,2)) where T<:Number
    nrow, ncol = size(x)
    W3d = hacdwt(x,wt,Lcol)
    W4d = Array{T,4}(undef,Lcol+1,ncol,Lrow+1,nrow)
    for i in 1:Lcol+1
        @inbounds W4d[i,:,:,:] = vacdwt(W3d[:,i,:],wt,Lrow)
    end
    W4d = permutedims(W4d, [4,2,3,1])
    return W4d
end

"""
    iacdwt(xw[, wt])

Performs the inverse autocorrelation discrete wavelet transform. Can be used for both the 1D
and 2D case.

!!! note
    The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `dwt` and `sdwt`, but is ignored during the reconstruction of signals.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number` or `xw::AbstractArray{T,4}`: ACDWT-transformed
  array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`::Vector{T}`: Inverse transformed signal.

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
function iacdwt(xw::AbstractArray{<:Number,2}, wt::Union{OrthoFilter,Nothing} = nothing)
    # Setup
    _, k = size(xw)
    v = xw[:,1]
    L = k-1

    # IACDWT
    for d in reverse(0:(L-1))
        w₁ = copy(v)
        @inbounds w₂ = @view xw[:,L-d+1]
        @inbounds iacdwt_step!(v, w₁, w₂)
    end
    return v
end

function iacdwt(xw::AbstractArray{T,4}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                T<:Number
    nrow, ncol, _, Lcol = size(xw)
    W4d = permutedims(xw,[4,2,3,1])
    W3d = Array{T,3}(undef, nrow, Lcol, ncol)
    for i in 1:Lcol
        for j in 1:nrow
            @inbounds W3d[j,i,:] = iacdwt(W4d[i,:,:,j])
        end
    end
    y = Array{T,2}(undef, nrow, ncol)
    for i in 1:ncol
        @inbounds y[:,i] = iacdwt(W3d[:,:,i])
    end
    return y
end

# ========== Autocorrelation Wavelet Packet Transform ==========
@doc raw"""
    acwpt(x, wt[, L])

Computes `L` levels of autocorrelation wavelet packet transforms (ACWPT) on `x`.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from ACWPT on `x`.

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
function acwpt(x::AbstractVector{T},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(x) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))

    # Setup
    Pmf, Qmf = make_acreverseqmfpair(wt)
    n = length(x)
    xw = Matrix{T}(undef, (n, 1<<L))
    xw[:,1] = x

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

"""
    iacwpt(xw[, wt])

Computes the inverse autocorrelation wavelet packet transform (IACWPT) on `xw`.

!!! note 
    The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `wpt` and `swpt`, but is ignored during the reconstruction of signals.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: ACWPT-transformed array.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
`::Vector{T}`: Inverse transformed signal.

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
function iacwpt(xw::AbstractArray{<:Number,2}, wt::Union{OrthoFilter,Nothing} = nothing)
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
            j₁ = (2*b)*nc + 1               # Parent and (scaling) child index
            j₂ = (2*b+1)*nc + 1             # Detail child index
            @inbounds v = @view temp[:,j₁]  # Parent node
            @inbounds w₁ = temp[:,j₁]       # Scaling node
            @inbounds w₂ = @view temp[:,j₂] # Detail node
            # Overwrite output of iSWPT directly onto temp
            @inbounds iacdwt_step!(v, w₁, w₂)
        end
    end
    return temp[:,1]
end

# ========== Autocorrelation Wavelet Packet Decomposition ==========
@doc raw"""
    acwpd(x, wt[, L])

Performs a discrete autocorrelation wavelet packet transform for a given signal `x`.
The wavelet type `wt` determines the transform type. Refer to Wavelet.jl for a list of available methods.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Original signal, preferably of size 2ᴷ where ``K
  \in \mathbb{N}``.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition.

# Returns
`::Matrix{T}`: Output from ACWPD on `x`.

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
function acwpd(x::AbstractVector{T}, 
               wt::OrthoFilter, 
               L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    # Setup
    Pmf, Qmf = make_acreverseqmfpair(wt)
    n = length(x)                       # Size of signal
    n₀ = 1<<(L+1)-1                     # Total number of nodes
    n₁ = n₀ - (1<<L)                    # Total number of nodes excluding leaf nodes
    xw = Matrix{T}(undef, (n, n₀))      # Output allocation
    xw[:,1] = x

    # ACWPD
    for i in 1:n₁
        d = floor(Int, log2(i))
        j₁ = left(i)
        j₂ = right(i)
        @inbounds v = @view xw[:,i]
        @inbounds w₁ = @view xw[:,j₁]
        @inbounds w₂ = @view xw[:,j₂]
        @inbounds acdwt_step!(w₁, w₂, v, d, Qmf, Pmf)
    end
    return xw
end

"""
    iacwpd(xw, L)
    iacwpd(xw[, wt, L])
    iacwpd(xw, tree)
    iacwpd(xw, wt, tree)

Performs the inverse autocorrelation discrete wavelet packet transform, with respect to a
decomposition tree.

!!! note The inverse autocorrelation transform does not require any wavelet filter, but an
    optional `wt` positional argument is included for the standardization of syntax with
    `wpt` and `swpt`, but is ignored during the reconstruction of signals.

!!! note This function might not be very useful if one is looking to reconstruct a raw
    decomposed signal. The purpose of this function would be better utilized in applications
    such as denoising, where a signal is decomposed (`swpd`) and thresholded
    (`denoise`/`denoiseall`) before being reconstructed.

# Arguments
- `xw::AbstractArray{T,2} where T<:Number`: ACWPD-transformed array.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of decomposition used
  for reconstruction.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly. 

# Returns
`::Vector{T}`: Inverse transformed signal.

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
```

**See also:** [`acwpd`](@ref)
"""
function iacwpd(xw::AbstractArray{T,2},
                wt::Union{OrthoFilter, Nothing} = nothing,
                L::Integer = maxtransformlevels(xw,1)) where T<:Number
    return iacwpd(xw, L)
end

function iacwpd(xw::AbstractArray{T,2}, L::Integer) where T<:Number
    # Sanity check
    @assert L ≤ maxtransformlevels(xw,1) ||
            throw(ArgumentError("Too many transform levels (length(x) < 2ᴸ)"))
    @assert L ≥ 1 || throw(ArgumentError("L must be ≥ 1"))
    return iacwpd(xw, maketree(size(xw,1), L, :full))
end

function iacwpd(xw::AbstractArray{T,2},
                wt::Union{OrthoFilter, Nothing},
                tree::BitVector) where T<:Number
    return iacwpd(xw, tree)
end

function iacwpd(xw::AbstractArray{T,2}, tree::BitVector) where T<:Number
    # Sanity check
    @assert isvalidtree(xw[:,1], tree)

    # Setup
    tmp = copy(xw)

    # iACWPD
    for (i, haschild) in Iterators.reverse(enumerate(tree))
        # Current node has child => Compute one step of inverse transform
        if haschild
            d = floor(Int, log2(i))         # Parent depth
            j₁ = left(i)                    # Scaling child index
            j₂ = right(i)                   # Detail child index
            @inbounds v = @view tmp[:,i]
            @inbounds w₁ = @view tmp[:,j₁]  # Scaling child node
            @inbounds w₂ = @view tmp[:,j₂]  # Detail child node
            # Inverse transform
            @inbounds iacdwt_step!(v, w₁, w₂)
        end
    end
    return tmp[:,1]
end

# function iacwpt(xw::AbstractArray{<:Number,2}, tree::BitVector)
#   @assert isvalidtree(xw[:,1], tree)
#   v₀ = iacwpt(xw,tree,2)
#   v₁ = iacwpt(xw,tree,3)
#   return (v₀ + v₁) / √2
# end

end # end module