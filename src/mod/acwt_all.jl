# ========== Transforms On A Group Of Signals ==========
# ----- ACDWT on a set of signals -----
"""
    acdwtall(x, wt[, L])

Computes the autocorrelation discrete wavelet transform (ACDWT) on each slice of signal.

# Arguments
- `x::AbstractArray{T,2} where T<:Number`: Input signals, where each column corresponds to a
  signal.
- `wt::OrthoFilter`: Orthogonal wavelet filter.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms.

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
function acdwtall(x::AbstractArray{T,2},
                  wt::OrthoFilter,
                  L::Integer = maxtransformlevels(x,1)) where T<:Number
    # Allocate space for transforms
    n,N = size(x)
    xw = Array{T,3}(undef, (n,L+1,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
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
function iacdwtall(xw::AbstractArray{T,3}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_x = 2
    dim_xw = 3
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
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms.

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
function acwptall(x::AbstractArray{T,2}, 
                  wt::OrthoFilter, 
                  L::Integer = maxtransformlevels(x,1)) where T<:Number
    # Allocate space for transforms
    n,N = size(x)
    xw = Array{T,3}(undef, (n,1<<L,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
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
function iacwptall(xw::AbstractArray{T,3}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
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
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms.

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
function acwpdall(x::AbstractArray{T,2},
                  wt::OrthoFilter,
                  L::Integer = maxtransformlevels(x,1)) where T<:Number
    # Allocate space for transforms
    n,N = size(x)
    xw = Array{T,3}(undef, (n,1<<(L+1)-1,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
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
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms.
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
function iacwpdall(xw::AbstractArray{T,3},
                   wt::Union{OrthoFilter,Nothing} = nothing,
                   L::Integer = maxtransformlevels(x,1)) where T<:Number
    return iacwpdall(xw, L)
end

function iacwpdall(xw::AbstractArray{T,3}, L::Integer) where T<:Number
    return iacwpdall(xw, maketree(size(xw,1), L, :full))
end

function iacwpdall(xw::AbstractArray{T,3},
                   wt::Union{OrthoFilter, Nothing},
                   tree::BitVector) where T<:Number
    return iacwpdall(xw, tree)
end

function iacwpdall(xw::AbstractArray{T,3}, tree::BitVector) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iacwpd!(xᵢ, xwᵢ)
        end
    end
    return x
end