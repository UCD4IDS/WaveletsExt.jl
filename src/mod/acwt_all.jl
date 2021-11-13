# ========== Transforms On A Group Of Signals ==========
# ----- ACDWT on a set of signals -----
"""
    acdwtall(x, wt[, L])

Computes the autocorrelation discrete wavelet transform (ACDWT) on each slice of signal.

# Arguments
- `x::AbstractArray{T,2} where T<:Number`: Input signals, where each column corresponds to a
  signal.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `minimum(size(xw)[1:end-1]) |> maxtransformlevels`) Number of
  levels of wavelet transforms.

# Returns
- `::Array{T,3}`: Slices of transformed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACDWT on all signals in x
xw = acdwtall(x, wt)
```

**See also:** [`acdwt`](@ref)
"""
function acdwtall(x::AbstractArray{T},
                  wt::OrthoFilter,
                  L::Integer = minimum(size(x)[1:end-1]) |> maxtransformlevels) where 
                  T<:Number
    @assert 2 ≤ ndims(x) ≤ 3
    # Allocate space for transforms
    sz = size(x)[1:(end-1)]
    k = ndims(x)==2 ? L+1 : 3*L+1
    N = size(x)[end]
    xw = Array{T}(undef, (sz...,k,N))
    # Dimension to slice
    dim_xw = ndims(xw)
    dim_x = ndims(x)
    # Compute transforms
    @inbounds begin
        @views for (xwᵢ, xᵢ) in zip(eachslice(xw, dims=dim_xw), eachslice(x, dims=dim_x))
            acdwt!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    iacdwtall(xw[, wt])

Computes the inverse autocorrelation discrete wavelet transform (IACDWT) on each slice of
signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: ACDWT-transformed signal.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACDWT on all signals in x
xw = acdwtall(x, wt)

# IACDWT on all signals in xw
x̂ = iacdwtall(xw)
```

**See also:** [`iacdwt`](@ref)
"""
function iacdwtall(xw::AbstractArray{T}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    @assert 3 ≤ ndims(xw) ≤ 4
    # Allocate space for transforms
    sz = size(xw)[1:(end-2)]
    N = size(xw)[end]
    x = Array{T}(undef, (sz...,N))
    # Dimension to slice
    dim_x = ndims(x)
    dim_xw = ndims(xw)
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iacdwt!(xᵢ, xwᵢ)
        end
    end
    return x
end

# ----- ACWPT on a set of signals -----
"""
    acwptall(x, wt[, L])

Computes the autocorrelation wavelet packet transform (ACWPT) on each slice of signal.

# Arguments
- `x""AbstractArray{T,2} where T<:Number`: Input signals, where each column corresponds to a
  signal.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `minimum(size(xw)[1:end-1]) |> maxtransformlevels`) Number of
  levels of wavelet transforms.

# Returns
- `::Array{T,3}`: Slices of transformed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACWPT on all signals in x
xw = acwptall(x, wt)
```

**See also:** [`acwpt`](@ref)
"""
function acwptall(x::AbstractArray{T}, 
                  wt::OrthoFilter, 
                  L::Integer = minimum(size(x)[1:end-1]) |> maxtransformlevels) where 
                  T<:Number
    @assert 2 ≤ ndims(x) ≤ 3
    # Allocate space for transforms
    sz = size(x)[1:(end-1)]
    k = ndims(x)==2 ? 1<<L : Int(4^L)
    N = size(x)[end]
    xw = Array{T}(undef, (sz...,k,N))
    # Dimension to slice
    dim_xw = ndims(xw)
    dim_x = ndims(x)
    # Compute transforms
    @inbounds begin
        @views for (xwᵢ, xᵢ) in zip(eachslice(xw, dims=dim_xw), eachslice(x, dims=dim_x))
            acwpt!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    iacwptall(xw[, wt])

Computes the inverse autocorrelation wavelet packet transform (IACWPT) on each slice of
signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: ACWPT-transformed signal.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACWPT on all signals in x
xw = acwptall(x, wt)

# IACWPT on all signals in xw
x̂ = iacwptall(xw)
```

**See also:** [`iacwpt`](@ref)
"""
function iacwptall(xw::AbstractArray{T}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    @assert 3 ≤ ndims(xw) ≤ 4
    # Allocate space for transforms
    sz = size(xw)[1:(end-2)]
    N = size(xw)[end]
    x = Array{T}(undef, (sz...,N))
    # Dimension to slice
    dim_x = ndims(x)
    dim_xw = ndims(xw)
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iacwpt!(xᵢ, xwᵢ)
        end
    end
    return x
end

# ----- ACWPD on a set of signals -----
"""
    acwpdall(x, wt[, L])

Computes the autocorrelation wavelet packet decomposition (ACWPD) on each slice of signal.

# Arguments
- `x""AbstractArray{T,2} where T<:Number`: Input signals, where each column corresponds to a
  signal.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `minimum(size(xw)[1:end-1]) |> maxtransformlevels`) Number of
  levels of wavelet transforms.

# Returns
- `::Array{T,3}`: Slices of transformed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACWPD on all signals in x
xw = acwpdall(x, wt)
```

**See also:** [`acwpd`](@ref)
"""
function acwpdall(x::AbstractArray{T},
                  wt::OrthoFilter,
                  L::Integer = minimum(size(x)[1:end-1]) |> maxtransformlevels) where 
                  T<:Number
    @assert 2 ≤ ndims(x) ≤ 3
    # Allocate space for transforms
    sz = size(x)[1:(end-1)]
    k = ndims(x)==2 ? 1<<(L+1)-1 : sum(4 .^(0:L))
    N = size(x)[end]
    xw = Array{T}(undef, (sz...,k,N))
    # Dimension to slice
    dim_xw = ndims(xw)
    dim_x = ndims(x)
    # Compute transforms
    @inbounds begin
        @views for (xwᵢ, xᵢ) in zip(eachslice(xw, dims=dim_xw), eachslice(x, dims=dim_x))
            acwpd!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    iacwpdall(xw[, wt, L])
    iacwpdall(xw, L)
    iacwpdall(xw, wt, tree)
    iacwpdall(xw, tree)

Computes the inverse autocorrelation wavelet packet decomposition (IACWPD) on each slice of
signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: ACWPT-transformed signal.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.
- `L::Integer`: (Default: `minimum(size(xw)[1:end-2]) |> maxtransformlevels`) Number of
  levels of wavelet transforms.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# ACWPD on all signals in x
xw = acwpdall(x, wt)

# IACWPD on all signals in xw
x̂ = iacwpdall(xw)
x̂ = iacwpdalll(xw, maketree(x))
x̂ = iacwpdall(xw, 5)
```

**See also:** [`iacwpd`](@ref)
"""
function iacwpdall(xw::AbstractArray{T},
                   wt::Union{OrthoFilter,Nothing} = nothing,
                   L::Integer = minimum(size(xw)[1:end-2]) |> maxtransformlevels) where 
                   T<:Number
    return iacwpdall(xw, L)
end

function iacwpdall(xw::AbstractArray{T}, L::Integer) where T<:Number
    return iacwpdall(xw, maketree(size(xw)[1:(end-2)]..., L))
end

function iacwpdall(xw::AbstractArray{T},
                   wt::Union{OrthoFilter, Nothing},
                   tree::BitVector) where T<:Number
    return iacwpdall(xw, tree)
end

function iacwpdall(xw::AbstractArray{T}, tree::BitVector) where T<:Number
    @assert 3 ≤ ndims(xw) ≤ 4
    # Allocate space for transforms
    sz = size(xw)[1:(end-2)]
    N = size(xw)[end]
    x = Array{T}(undef, (sz...,N))
    # Dimension to slice
    dim_x = ndims(x)
    dim_xw = ndims(xw)
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iacwpd!(xᵢ, xwᵢ, tree)
        end
    end
    return x
end