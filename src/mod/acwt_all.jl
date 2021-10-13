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
```
"""
function iacdwtall(xw::AbstractArray{T,3}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    # Allocate space for transforms
    n,_,N = size(x)
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

"""
function iacwptall(xw::AbstractArray{T,3}, wt::Union{OrthoFilter,Nothing} = nothing) where 
                   T<:Number
    # Allocate space for transforms
    n,_,N = size(x)
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
    n,_,N = size(x)
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