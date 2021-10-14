# ========== Transforms On A Group Of Signals ==========
# ----- SDWT on a set of signals -----
"""
    sdwtall(x, wt[, L])

Computes the stationary discrete wavelet transform (SDWT) on each slice of signal.

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

# SDWT on all signals in x
xw = sdwtall(x, wt)
```

**See also:** [`sdwt`](@ref)
"""
function sdwtall(x::AbstractArray{T,2},
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
            sdwt!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    isdwtall(xw[, wt])
    isdwtall(xw, wt, sm)

Computes the inverse stationary discrete wavelet transform (ISDWT) on each slice of signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: SDWT-transformed signal.
- `wt::OrthoFilter`: (Default: `nothing`) Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# SDWT on all signals in x
xw = sdwtall(x, wt)

# ISDWT on all signals in xw
x̂ = isdwtall(xw)
```

**See also:** [`isdwt`](@ref)
"""
function isdwtall(xw::AbstractArray{T,3}, wt::OrthoFilter) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_x = 2
    dim_xw = 3
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            isdwt!(xᵢ, xwᵢ, wt)
        end
    end
    return x
end

function isdwtall(xw::AbstractArray{T,3}, wt::OrthoFilter, sm::Integer) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_x = 2
    dim_xw = 3
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            isdwt!(xᵢ, xwᵢ, wt, sm)
        end
    end
    return x
end

# ----- SWPT on a set of signals -----
"""
    swptall(x, wt[, L])

Computes the stationary wavelet packet transform (SWPT) on each slice of signal.

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

# SWPT on all signals in x
xw = swptall(x, wt)
```

**See also:** [`swpt`](@ref)
"""
function swptall(x::AbstractArray{T,2}, 
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
            swpt!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    iswptall(xw[, wt])
    iswptall(xw, wt, sm)

Computes the inverse stationary wavelet packet transform (ISWPT) on each slice of signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: SWPT-transformed signal.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# SWPT on all signals in x
xw = swptall(x, wt)

# ISWPT on all signals in xw
x̂ = iswptall(xw)
```

**See also:** [`iswpt`](@ref)
"""
function iswptall(xw::AbstractArray{T,3}, wt::OrthoFilter) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iswpt!(xᵢ, xwᵢ, wt)
        end
    end
    return x
end

function iswptall(xw::AbstractArray{T,3}, wt::OrthoFilter, sm::Integer) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iswpt!(xᵢ, xwᵢ, wt, sm)
        end
    end
    return x
end

# ----- SWPD on a set of signals -----
"""
    swpdall(x, wt[, L])

Computes the stationary wavelet packet decomposition (SWPD) on each slice of signal.

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

# SWPD on all signals in x
xw = swpdall(x, wt)
```

**See also:** [`swpd`](@ref)
"""
function swpdall(x::AbstractArray{T,2},
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
            swpd!(xwᵢ, xᵢ, wt, L)
        end
    end
    return xw
end

"""
    iswpdall(xw, wt, L, sm)
    iswpdall(xw, wt[, L])
    iswpdall(xw, wt, tree, sm)
    iswpdall(xw, wt, tree)

Computes the inverse autocorrelation wavelet packet decomposition (ISWPD) on each slice of
signal.

# Arguments
- `xw::AbstractArray{T,3} where T<:Number`: SWPT-transformed signal.
- `wt::Union{OrthoFilter, Nothing}`: (Default: `nothing`) Orthogonal wavelet filter.
- `L::Integer`: (Default: `Wavelets.maxtransformlevels(xᵢ)`) Number of levels of wavelet
  transforms.
- `tree::BitVector`: Binary tree for inverse transform to be computed accordingly.
- `sm::Integer`: If `sm` is included as an argument, the `sm`-shifted inverse transform will
  be computed. This results in significantly faster computation, but fails to fully utilize
  the strength of redundant wavelet transforms.

# Returns
- `::Array{T,2}`: Slices of reconstructed signals.

# Examples
```julia
using Wavelets, WaveletsExt

# Generate random signals
x = randn(32, 5)
# Create wavelet
wt = wavelet(WT.db4)

# SWPD on all signals in x
xw = swpdall(x, wt)

# ISWPD on all signals in xw
x̂ = iswpdall(xw)
x̂ = iswpdalll(xw, maketree(x))
x̂ = iswpdall(xw, 5)
```

**See also:** [`iswpd`](@ref)
"""
function iswpdall(xw::AbstractArray{T,3},
                   wt::OrthoFilter,
                   L::Integer = maxtransformlevels(x,1)) where T<:Number
    return iswpdall(xw, wt, maketree(size(xw,1), L))
end

function iswpdall(xw::AbstractArray{T,3}, wt::OrthoFilter, L::Integer, sm::Integer) where 
                   T<:Number
    return iswpdall(xw, wt, maketree(size(xw,1), L), sm)
end

function iswpdall(xw::AbstractArray{T,3}, wt::OrthoFilter, tree::BitVector) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iswpd!(xᵢ, xwᵢ, wt, tree)
        end
    end
    return x
end

function iswpdall(xw::AbstractArray{T,3}, 
                  wt::OrthoFilter, 
                  tree::BitVector, 
                  sm::Integer) where T<:Number
    # Allocate space for transforms
    n,_,N = size(xw)
    x = Array{T,2}(undef, (n,N))
    # Dimension to slice
    dim_xw = 3
    dim_x = 2
    # Compute transforms
    @inbounds begin
        @views for (xᵢ, xwᵢ) in zip(eachslice(x, dims=dim_x), eachslice(xw, dims=dim_xw))
            iswpd!(xᵢ, xwᵢ, wt, tree, sm)
        end
    end
    return x
end