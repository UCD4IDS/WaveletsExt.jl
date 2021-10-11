module ACWT
export
    # Autocorrelation wavelet transform
    acdwt, 
    acwpt,
    acwpd,
    # Inverse autocorrelation wavelet transform
    iacdwt,
    iacwpt,
    iacwpd, 
    # Functions to be deprecated
    acwt,
    iacwt

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

Returns a tuple `(v, w)` of the scaling and detail coefficients at level `j+1`.

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

function iacdwt_step(w₁::AbstractVector{T}, w₂::AbstractVector{T}) where T<:Number
    v = similar(w₁)
    iacdwt_step!(v, w₁, w₂)
    return v
end

function iacdwt_step!(v::AbstractVector{T}, 
                      w₁::AbstractVector{T}, 
                      w₂::AbstractVector{T}) where T<:Number
    @assert length(v) == length(w₁) == length(w₂)
    for i in eachindex(v)
        @inbounds v[i] = (w₁[i]+w₂[i]) / √2
    end
end

# ========== Autocorrelation Discrete Wavelet Transform ==========
"""
	acdwt(x, wt[, L=maxtransformlevels(x)])
    acdwt(x, wt[, Lrow=maxtransformlevels(x[1,:]), Lcol=maxtransformlevels(x[:,1])])

Performs a discrete autocorrelation wavelet transform for a given signal `x`.
The signal can be 1D or 2D. The wavelet type `wt` determines the transform type.
Refer to Wavelet.jl for a list of available methods.

# Examples
```julia
acdwt(x, wavelet(WT.db4))

acdwt(x, wavelet(WT.db4), 4) # level 4 decomposition
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
	iacdwt(xw::AbstractArray{<:Number,2})
    iacdwt(xw::AbstractArray{<:Number,4})

Performs the inverse autocorrelation discrete wavelet transform. 
Can be used for both the 1D and 2D case.

**See also:** [`iacdwt!`](@ref), [`acdwt`](@ref)
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

function iacdwt(xw::AbstractArray{T,4}) where T<:Number
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
"""
    acwpd(x, wt[, L=maxtransformlevels(x)])

Performs a discrete autocorrelation wavelet packet transform for a given signal `x`.
The wavelet type `wt` determines the transform type. Refer to Wavelet.jl for a list of available methods.

# Examples
```julia
acwpd(x, wavelet(WT.db4))

acwpd(x, wavelet(WT.db4), 4)
```

**See also:** [`acdwt`](@ref), [`acwpt_step`](@ref), [`iacwpt`](@ref)
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
    iacwpd(xw, tree, i)

Performs the inverse autocorrelation discrete wavelet packet transform,
with respect to a decomposition tree.

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

# ========== Deprecated Functions ==========================================================
# Functions that will be completely deleted by v0.2.0
function acwt(args...)
    Base.depwarn("`acwt` is deprecated, use `acdwt` instead.", :acwt, force=true)
    return acdwt(args...)
end

function iacwt(args...)
    Base.depwarn("`iacwt` is deprecated, use `iacdwt` instead.", :iacwt, force=true)
    return iacdwt(args...)
end

function vacwt(args...)
    Base.depwarn("`vacwt` is deprecated, use `vacdwt` instead.", :acwt, force=true)
    return vacdwt(args...)
end

function hacwt(args...)
    Base.depwarn("`hacwt` is deprecated, use `hacdwt` instead.", :acwt, force=true)
    return hacdwt(args...)
end

function acwpt(args...)
    Base.depwarn("`acwpt` is deprecated, use `acwpd` instead.", :acwpt, force=true)
    return acwpd(args...)
end

end # end module