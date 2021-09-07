module WPD
export 
    wpd,
    wpd!

using 
    Wavelets

using 
    ..Utils

"""
    wpd(x, wt[, L=maxtransformlevels(x)])

    wpd(x, wt, hqf, gqf[, L=maxtransformlevels(x)])

Returns the wavelet packet decomposition WPD) for L levels for input signal(s) 
x.

**See also:** [`wpd!`](@ref)
"""
function wpd(x::AbstractVector{<:Number}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x))

    @assert 0 <= L <= maxtransformlevels(x)
    gqf, hqf = WT.makereverseqmfpair(wt, true)
    
    return wpd(x, wt, hqf, gqf, L)
end

function wpd(x::AbstractVector{T}, wt::DiscreteWavelet, hqf::Array{S,1}, 
        gqf::Array{S,1}, L::Integer=maxtransformlevels(x)) where 
        {T<:Number, S<:Number}

    n = length(x)
    @assert isdyadic(n)
    @assert 0 <= L <= maxtransformlevels(n)

    result = Array{T, 2}(undef, (n, L+1))
    wpd!(result, x, wt, hqf, gqf, L)
    return result
end


"""
    wpd!(y, x, wt[, L=maxtransformlevels(x)])

    wpd!(y, x, wt, hqf, gqf[, L=maxtransformlevels(x)])

Same as `wpd` but without array allocation.

**See also:** [`wpd`](@ref)
"""
function wpd!(y::AbstractArray{T,2}, x::AbstractArray{T,1},
        wt::DiscreteWavelet, L::Integer=maxtransformlevels(x)) where
        T<:Number

    gqf, hqf = WT.makereverseqmfpair(wt, true)
    wpd!(y, x, wt, hqf, gqf, L)
    return nothing
end

function wpd!(y::AbstractArray{T,2}, 
              x::AbstractArray{T,1}, 
              wt::DiscreteWavelet, 
              hqf::Array{S,1}, 
              gqf::Array{S,1}, 
              L::Integer=maxtransformlevels(x)) where 
             {T<:Number, S<:Number}
    n = length(x)
    @assert 0 <= L <= maxtransformlevels(n)
    @assert size(y) == (n, L+1)

    y[:,1] = x
    si = Vector{S}(undef, length(wt)-1)
    for i in 0:(L-1)
        # parent node length
        nₚ = nodelength(n, i) 
        for j in 0:((1<<i)-1)
            # extract parent node
            colₚ = i + 1
            rng = (j * nₚ + 1):((j + 1) * nₚ)
            @inbounds nodeₚ = @view y[rng, colₚ]

            # extract left and right child nodes
            colₘ = colₚ + 1
            @inbounds nodeₘ = @view y[rng, colₘ]

            # perform 1 level of wavelet decomposition
            Transforms.unsafe_dwt1level!(nodeₘ, nodeₚ, wt, true, hqf, gqf, si)
        end
    end
    return nothing
end

# ========== Transforms On A Group Of Signals ==========
"""
    dwtall(x, wt[, L])

Computes the discrete wavelet transform (DWT) on each slice of signal. Signals are sliced on
the ``n``-th dimension for an ``n``-dimensional input `x`.

*Note:* `dwt` is currently available for 1-D, 2-D, and 3-D signals only. 

# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to
  one signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms. 

# Returns
`::Array{T}`: Slices of transformed signals. Signals are sliced the same way as the input
signal `x`.

# Examples
```
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# DWT on all signals in x
xw = dwtall(x, wt)
```
"""
function dwtall(x::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(x) > 1

    # Allocate space for transforms
    y = similar(x)
    # Dimension to slice
    dim = ndims(y)
    # Compute transforms
    @views for (yᵢ, xᵢ) in zip(eachslice(y, dims=dim), eachslice(x, dims=dim))
        dwt!(yᵢ, xᵢ, args...)
    end
    return y
end

"""
    wptall(x, wt)

    wptall(x, wt, L)

    wptall(x, wt, tree)

Computes the wavelet packet transform (WPT) on each slice of signal. Signals are sliced on
the ``n``-th dimension for an ``n``-dimensional input `x`.

*Note:* `wpt` is currently available for 1-D signals only.
    
# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to
  one signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  decomposition. 
- `tree::BitVector`: (Default: `Wavelets.maketree(xᵢ, :full)`) Tree to follow for wavelet
  decomposition. 

# Returns
`::Array{T}`: Slices of transformed signals. Signals are sliced the same way as the input
signal `x`.

# Examples
```
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# WPT on all signals in x
xw = wptall(x, wt)
```
"""
function wptall(x::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(x) > 1
    
    # Allocate space for transform
    y = similar(x)
    # Dimension to slice
    dim = ndims(x)
    # Compute transforms
    @views for (yᵢ, xᵢ) in zip(eachslice(y, dims=dim), eachslice(x, dims=dim))
        wpt!(yᵢ, Array(xᵢ), args...)
    end
    return y
end

"""
    wpdall(x, wt[, L])

    wpdall(y, x, wt, hqf, gqf[, L])

Computes the wavelet packet decomposition (WPD) on each slice of signal. Signals are sliced
on the ``n``-th dimension for an ``n``-dimensional input `x`.

*Note:* `wpd` is currently available for 1-D signals only.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to
  one signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  decomposition. 

# Returns
`::Array{T}`
"""
function wpdall(x::AbstractArray{T}, args...) where T<:Number
    # TODO: Not the most optimal solution. Using list comprehension + concatenating the list
    # TODO: of arrays means array allocation is done on the fly, ie. more arrays to
    # TODO: decompose = more allocations = more time wasted. Doing concatenation after that
    # TODO: further wastes more time.
    # Sanity check
    @assert ndims(x) > 1
        
    # Dimension to slice
    dim = ndims(x)
    # Compute transforms
    y = [wpd(xᵢ, args...) for xᵢ in eachslice(x, dims=dim)] |>
        xw -> cat(xw..., dims=dim+1)
    return y
end

end # end module