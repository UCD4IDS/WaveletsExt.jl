# ========== Transforms On A Group Of Signals ==========
# ----- Discrete Wavelet Transform on a set of signals -----
"""
    dwtall(x, wt[, L])

Computes the discrete wavelet transform (DWT) on each slice of signal. Signals are sliced on
the ``n``-th dimension for an ``n``-dimensional input `x`.

!!! note
    `dwt` is currently available for 1-D, 2-D, and 3-D signals only. 

# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to
  one signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms. 

# Returns
`::Array{T}`: Slices of transformed signals. Signals are sliced the same way as the input
signal `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# DWT on all signals in x
xw = dwtall(x, wt)
```

**See also:** [`idwtall`](@ref), [`wpdall`](@ref), [`wptall`](@ref)
"""
function dwtall(x::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(x) > 1

    # Allocate space for transforms
    y = similar(x)
    # Dimension to slice
    dim = ndims(y)
    # Compute transforms
    @inbounds begin
        @views for (yᵢ, xᵢ) in zip(eachslice(y, dims=dim), eachslice(x, dims=dim))
            dwt!(yᵢ, xᵢ, args...)
        end
    end
    return y
end

"""
    idwtall(xw, wt[, L])

Computes the inverse discrete wavelet transform (iDWT) on each slice of signal. Signals are
sliced on the ``n``-th dimension for an ``n``-dimensional input `xw`.

!!! note
    `idwt` is currently available for 1-D, 2-D, and 3-D signals only. 

# Arguments
- `xw::AbstractArray{T} where T<:Number`: Input decomposed signals, where each slice
  corresponds to one signal. For a set of input signals `xw` of dimension ``n``, signals are
  sliced on the ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xwᵢ)`) Number of levels of wavelet
  transforms. 

# Returns
`::Array{T}`: Slices of reconstructed signals. Signals are sliced the same way as the input
`xw`.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# DWT on all signals in x
xw = dwtall(x, wt)

# iDWT on all signals
x̂ = idwtall(xw, wt)
```

**See also:** [`dwtall`](@ref), [`iwpdall`](@ref), [`iwptall`](@ref)
"""
function idwtall(xw::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(xw) > 1

    # Setup
    y = similar(xw)      # Allocation for output
    dim = ndims(xw)      # Dimension to slice

    # IDWT
    @inbounds begin
        @views for (yᵢ, xwᵢ) in zip(eachslice(y, dims=dim), eachslice(xw, dims=dim))
            idwt!(yᵢ, xwᵢ, args...)
        end
    end
    return y
end

# ----- Wavelet Packet Transforms on a set of signals -----
"""
    wptall(x, wt[, L])
    wptall(x, wt, tree)

Computes the wavelet packet transform (WPT) on each slice of signal. Signals are sliced on
the ``n``-th dimension for an ``n``-dimensional input `x`.

!!! note
    `wpt` is currently available for 1-D and 2-D signals only.
    
# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to
  one signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  decomposition. 
- `tree::BitVector`: (Default: `Wavelets.maketree(xᵢ, :full)`) Tree to follow for wavelet
  decomposition. Default value is only applicable for 1D signals.

# Returns
`::Array{T}`: Slices of transformed signals. Signals are sliced the same way as the input
signal `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# WPT on all signals in x
xw = wptall(x, wt)
```

**See also:** [`wpdall`](@ref), [`dwtall`](@ref), [`wpt`](@ref)
"""
function wptall(x::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(x) > 1
    
    # Setup
    y = similar(x)      # Allocation for output
    dim = ndims(x)      # Dimension to slice
    # Compute transforms
    @inbounds begin
        @views for (yᵢ, xᵢ) in zip(eachslice(y, dims=dim), eachslice(x, dims=dim))
            wpt!(yᵢ, Array(xᵢ), args...)
        end
    end
    return y
end

"""
    iwptall(xw, wt[, L])
    iwptall(xw, wt, tree)

Computes the inverse wavelet packet transform (iWPT) on each slice of signal. Signals are
sliced on the ``n``-th dimension for an ``n``-dimensional input `xw`.

!!! note 
    `iwpt` is currently available for 1-D and 2-D signals only.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to one
  signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  decomposition. 
- `tree::BitVector`: (Default: `Wavelets.maketree(xᵢ, :full)`) Tree to follow for wavelet
  decomposition. Default value is only applicable for 1D signals.

# Returns
`::Array{T}`: Slices of transformed signals. Signals are sliced the same way as the input
signal `x`.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# WPT on all signals in x
xw = wptall(x, wt)

# iWPT on all signals on x
x̂ = iwptall(xw, wt)
```

**See also:** [`wpdall`](@ref), [`dwtall`](@ref), [`wpt`](@ref)
"""
function iwptall(xw::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(xw) > 1

    # Setup
    y = similar(xw)      # Allocation for output
    dim = ndims(xw)      # Dimension to slice

    # IWPT
    @inbounds begin
        @views for (yᵢ, xwᵢ) in zip(eachslice(y, dims=dim), eachslice(xw, dims=dim))
            iwpt!(yᵢ, Array(xwᵢ), args...)
        end
    end
    return y
end

# ----- Wavelet Packet Decomposition on a set of signals -----
"""
    wpdall(x, wt[, L])

Computes the wavelet packet decomposition (WPD) on each slice of signal. Signals are sliced
on the ``n``-th dimension for an ``n``-dimensional input `x`.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to one
  signal. For a set of input signals `x` of dimension ``n``, signals are sliced on the
  ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `minimum(size(x)[1:end-1]) |> maxtransformlevels`) Number of
  levels of wavelet decomposition. 

# Returns
`::Array{T}`: Array of decomposed signals. Signals are sliced by the final dimension.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# WPT on all signals in x
xw = wpdall(x, wt)
```

**See also:** [`wptall`](@ref), [`dwtall`](@ref), [`wpd`](@ref), [`wpd!`](@ref)
"""
function wpdall(x::AbstractArray{T}, 
                wt::OrthoFilter, 
                L::Integer = minimum(size(x)[1:end-1]) |> maxtransformlevels) where 
                T<:Number
    # Sanity check
    @assert ndims(x) > 1
    @assert 0 ≤ L ≤ maxtransformlevels(selectdim(x, ndims(x), 1))
       
    # Allocate space for transform
    sz = size(x)[begin:end-1]  # Signal size
    N = size(x)[end]           # Number of signals
    y = Array{T}(undef, (sz...,L+1,N))
    # Dimension to slice
    dim_y = ndims(y)
    dim_x = ndims(x)
    # Compute transforms
    @inbounds begin
        @views for (yᵢ, xᵢ) in zip(eachslice(y, dims=dim_y), eachslice(x, dims=dim_x))
            wpd!(yᵢ, Array(xᵢ), wt, L)
        end
    end
    return y
end

"""
    iwpdall(xw, wt[, L; standard])
    iwpdall(xw, wt, tree[; standard])

Computes the inverse wavelet packet decomposition (IWPD) on each slice of signal. Signals
are sliced on the ``n``-th dimension for an ``n``-dimensional input `xw`.

# Arguments
- `xw::AbstractArray{T} where T<:Number`: Input signals, where each slice corresponds to one
  signal decomposition. For a set of input signals `x` of dimension ``n``, `xw` is sliced on
  the ``n``-th dimension.
- `wt::OrthoFilter`: Wavelet used.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  decomposition. 

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
`::Array{T}`: Array of decomposed signals. Signals are sliced by the final dimension.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# WPD on all signals in x
xw = wpdall(x, wt)

# IWPD on all signals
x̂ = iwpdall(xw, wt)
```

**See also:** [`wptall`](@ref), [`dwtall`](@ref), [`wpd`](@ref), [`wpd!`](@ref)
"""
function iwpdall(xw::AbstractArray{T}, args...) where T<:Number
    # Sanity check
    @assert ndims(xw) > 2

    # Setup
    sz = size(xw)[1:(end-2)]
    N = size(xw)[end]
    x̂ = Array{T}(undef, (sz...,N))
    # Dimension to slice
    dim_xw = ndims(xw)
    dim_x̂ = ndims(x̂)
    # Compute transforms
    @inbounds begin
        @views for (x̂ᵢ, xwᵢ) in zip(eachslice(x̂, dims=dim_x̂), eachslice(xw, dims=dim_xw))
            iwpd!(x̂ᵢ, Array(xwᵢ), args...)
        end
    end
    return x̂
end
