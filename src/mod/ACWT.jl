module ACWT
export
    acwt, 
    iacwt,
    acwpt, 
    iacwpt

using ..Utils
using LinearAlgebra, Wavelets

"""
	acwt(x, wt[, L=maxtransformlevels(x)])

    acwt(x, wt[, Lrow=maxtransformlevels(x[1,:]), Lcol=maxtransformlevels(x[:,1])])

Performs a discrete autocorrelation wavelet transform for a given signal `x`.
The signal can be 1D or 2D. The wavelet type `wt` determines the transform type.
Refer to Wavelet.jl for a list of available methods.

# Examples
```julia
acwt(x, wavelet(WT.db4))

acwt(x, wavelet(WT.db4), 4) # level 4 decomposition
```

**See also:** [`acwt_step`](@ref), [`iacwt`](@ref)
"""
function acwt end

"""
    acwt_step(v, j, h, g)

Performs one level of the autocorrelation discrete wavelet transform (ACWT) on the 
vector `v`, which is the j-th level scaling coefficients (Note the 0th level
scaling coefficients is the raw signal). The vectors `h` and `g` are the detail
and scaling filters.

Returns a tuple `(v, w)` of the scaling and detail coefficients at level `j+1`.

**See also:** [`acwt`](@ref), [`iacwt`](@ref)
"""
function acwt_step end

"""
    acwpt(x, wt[, L=maxtransformlevels(x)])

Performs a discrete autocorrelation wavelet packet transform for a given signal `x`.
The wavelet type `wt` determines the transform type. Refer to Wavelet.jl for a list of available methods.

# Examples
```julia
acwpt(x, wavelet(WT.db4))

acwpt(x, wavelet(WT.db4), 4)
```

**See also:** [`acwt`](@ref), `acwpt_step`, [`iacwpt`](@ref)
"""
function acwpt end

"""
    acwpt_step(W, i, d, Qmf, Pmf)

Performs one level of the autocorrelation discrete wavelet packet transform 
(ACWPT) on the `i`-th node at depth `d` in the array `W`. The vectors `Qmf` and 
`Pmf` are the detail and scaling filters.

**See also:** [`acwpt`](@ref), [`iacwpt`](@ref)
"""
function acwpt_step end

"""
	iacwt(xw::AbstractArray{<:Number,2})

    iacwt(xw::AbstractArray{<:Number,4})

Performs the inverse autocorrelation discrete wavelet transform. 
Can be used for both the 1D and 2D case.

**See also:** [`iacwt!`](@ref), [`acwt`](@ref)
"""
function iacwt end

"""
	iacwt!(xw::AbstractArray{<:Number,2})

Same as `iacwt` but performs the inverse autocorrelation discrete wavelet transform in place.

**See also:** [`iacwt`](@ref), [`acwt`](@ref)
"""
function iacwt! end

"""
    iacwpt(xw, tree, i)

Performs the inverse autocorrelation discrete wavelet packet transform,
with respect to a decomposition tree.

**See also:** [`acwpt`](@ref)
"""
function iacwpt end

### ACWT Base methods ###
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

**See also:** [`qfliter`](@ref)
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

Generates the low-pass autocorrelation filter

**See also:** [`pfliter`](@ref)
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

**See also:** [`make_acreverseqmfpair`](@ref)
"""
function make_acqmfpair(f::OrthoFilter)
    pmfilter, qmfilter = pfilter(f), qfilter(f)
    return pmfilter, qmfilter
end

"""
    make_acreverseqmfpair(f::OrthoFilter)

Generates the reverse autocorrelation quadratic mirror filters.

**See also:** [`make_acqmfpair`](@ref)
"""
function make_acreverseqmfpair(f::OrthoFilter)
    pmf, qmf = make_acqmfpair(f)
    return reverse(pmf), reverse(qmf)
end

### ACWT Transforms ### 
## 1D ##
function acwt_step(v::AbstractVector{T}, j::Integer, h::Array{T,1}, g::Array{T,1}) where {T <: Number}
    N = length(v)
    L = length(h)
    v1 = zeros(T, N)
    w1 = zeros(T, N)
  
    @inbounds begin
        for i in 1:N
            t = i
            i = mod1(i + (L÷2+1) * 2^(j-1),N) # Need to shift by half the filter size because of periodicity assumption 
            for n in 1:L
                t += 2^(j-1)
                t = mod1(t, N)
                w1[i] += h[n] * v[t]
                v1[i] += g[n] * v[t]
            end
        end
    end
    return v1, w1 
end

function acwt(x::AbstractVector{<:Number}, wt::OrthoFilter, L::Integer=maxtransformlevels(x))

    @assert L <= maxtransformlevels(x) || throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))
  
    # Setup
    n = length(x)
    Pmf, Qmf = make_acreverseqmfpair(wt)
    wp = zeros(n,L+1)
    wp[:,1] = x
  
    for j in 1:L
        @inbounds wp[:,1], wp[:,L+2-j] = acwt_step(wp[:,1],j,Qmf,Pmf)
    end
  
    return wp
end

## 2D ##
"""
    hacwt(x, wt[, L=maxtransformlevels(x)])

Computes the column-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`vacwt`](@ref)
"""
function hacwt(x::AbstractArray{<:Number,2}, wt::OrthoFilter, L::Integer=maxtransformlevels(x[:,1]))
    nrow, ncol = size(x)
    W = Array{Float64,3}(undef,nrow,L+1,ncol)
    for i in 1:ncol
        @inbounds W[:,:,i] = acwt(x[:,i],wt,L)
    end
    return W
end

"""
    vacwt(x, wt[, L=maxtransformlevels(x)])

Computes the row-wise discrete autocorrelation transform coeficients for 2D signals.

**See also:** [`hacwt`](@ref)
"""
function vacwt(x::AbstractArray{<:Number,2}, wt::OrthoFilter, L::Integer=maxtransformlevels(x[1,:]))
    nrow, ncol = size(x)
    W = Array{Number,3}(undef,ncol,L+1,nrow)
    for i in 1:nrow
        W[:,:,i] = acwt(x[i,:],wt,L)
    end
    return W
end

function acwt(x::AbstractArray{<:Number,2}, wt::OrthoFilter,
              Lrow::Integer=maxtransformlevels(x[1,:]),
              Lcol::Integer=maxtransformlevels(x[:,1]))
    nrow, ncol = size(x)
    W3d = hacwt(x,wt,Lcol)
    W4d = Array{Number,4}(undef,Lcol+1,ncol,Lrow+1,nrow)
    for i in 1:Lcol+1
        @inbounds W4d[i,:,:,:] = vacwt(W3d[:,i,:],wt,Lrow)
    end
    W4d = permutedims(W4d, [4,2,3,1])
    return W4d
end

## ACW Packet Transform ##
function acwpt_step(W::AbstractArray{T,2}, i::Integer, d::Integer, Qmf::Vector{T}, Pmf::Vector{T}) where T <: Number
    n,m = size(W)
    if i<<1+1 <= m
        W[:,i<<1], W[:,i<<1+1] = acwt_step(W[:,i],d,Qmf,Pmf)
        acwpt_step(W,i<<1,d+1,Qmf,Pmf) # left
        acwpt_step(W,i<<1+1,d+1,Qmf,Pmf) # right
    end
end

function acwpt(x::AbstractVector{T}, wt::OrthoFilter, L::Integer=maxtransformlevels(x)) where T<:Number
    W = Array{Float64,2}(undef,length(x),2<<L-1)
    W[:,1] = x
    Pmf, Qmf = ACWT.make_acreverseqmfpair(wt)
    acwpt_step(W,1,1,Qmf,Pmf)
    return W
end

### Inverse Transforms ###
function iacwt!(xw::AbstractArray{<:Number,2})
    n,m = size(xw)
    @inbounds begin
        for i = 2:m
            xw[:,1] = (xw[:,1] + xw[:,i]) / √2
        end
    end
end

function iacwt(xw::AbstractArray{<:Number,2})
    y = deepcopy(xw)
    iacwt!(y)
    return y[:,1]
end

function iacwt(xw::AbstractArray{<:Number,4})
    nrow, ncol, Lrow, Lcol = size(xw)
    W4d = permutedims(xw,[4,2,3,1])
    W3d = Array{Number,3}(undef, nrow, Lcol, ncol)
    for i in 1:Lcol
        for j in 1:nrow
            @inbounds W3d[j,i,:] = iacwt(W4d[i,:,:,j])
        end
    end
    y = Array{Number,2}(undef, nrow, ncol)
    for i in 1:ncol
        @inbounds y[:,i] = iacwt(W3d[:,:,i])
    end
    return y
end

function iacwpt(xw::AbstractArray{<:Number,2}, tree::BitVector, i::Integer=1)

    @assert i <= size(xw, 2)
    @assert isvalidtree(xw[:,1], tree)
    n₀ = length(tree)
    if i > n₀ || tree[i] == false      # leaf node 
        return xw[:,i]
    end

    v₀ = iacwpt(xw,tree,left(i))
    v₁ = iacwpt(xw,tree,right(i))

    return (v₀ + v₁) / √2
end

# function iacwpt(xw::AbstractArray{<:Number,2}, tree::BitVector)
#   @assert isvalidtree(xw[:,1], tree)
#   v₀ = iacwpt(xw,tree,2)
#   v₁ = iacwpt(xw,tree,3)
#   return (v₀ + v₁) / √2
# end

end # end module