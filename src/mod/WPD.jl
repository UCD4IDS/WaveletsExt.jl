module WPD
export 
    wpd,
    wpd!,
    dwtall,
    wptall,
    wpdall,
    idwtall,
    iwptall,
    iwpdall

using 
    Wavelets

using 
    ..Utils

# ========== Wavelet Packet Decomposition ==========
# 1D Wavelet Packet Decomposition without allocated output array
"""
    wpd(x, wt[, L])
    wpd(x, wt[, L; standard])

Returns the wavelet packet decomposition (WPD) for L levels for input signal x.

!!! note 
    Nonstandard transform is not yet available for 2D-WPD. Setting `standard = true` will 
    currently result in an error message.

# Arguments
- `x::AbstractVector{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet decomposition whereas a matrix input
  undergoes 2D wavelet decomposition.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,2}` or `::Array{T,3}`: Decomposed signal. For an input vector, output is a 2D
  matrix where each column corresponds to a level of decomposition. For an input matrix,
  output is a 3D array where each slice of dimension 3 corresponds to a level of
  decomposition.

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)

# 2D wavelet decomposition
x = randn(8,8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)
```

**See also:** [`wpd!`](@ref)
"""
function wpd(x::AbstractVector{T}, 
             wt::OrthoFilter, 
             L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert isdyadic(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)

    # Allocate variable for result
    n = length(x)
    xw = Array{T,2}(undef, (n,L+1))
    # Wavelet packet decomposition
    wpd!(xw, x, wt, L)
    return xw
end

# 2D Wavelet Packet Decomposition without allocated output array
function wpd(x::AbstractArray{T,2},
             wt::OrthoFilter,
             L::Integer = maxtransformlevels(x);
             standard::Bool = true) where T<:Number
    # Sanity check
    @assert 0 ≤ L ≤ maxtransformlevels(x)

    # Allocate variable for result
    sz = size(x)
    xw = Array{T,3}(undef, (sz...,L+1))
    # Wavelet packet decomposition
    wpd!(xw, x, wt, L, standard=standard)
    return xw
end

# 1D Wavelet Packet Decomposition with allocated output array
"""
    wpd!(y, x, wt[, L])
    wpd!(y, x, wt[, L; standard])

Same as `wpd` but without array allocation.

!!! note 
    Nonstandard transform is not yet available for 2D-WPD. Setting `standard = true` will 
    currently result in an error message.

# Arguments
- `y::AbstractArray{T,2} where T<:Number` or `y::AbstractArray{T,3} where T<:Number`: An
  allocated array to write the outputs of `x` onto.
- `x::AbstractVector{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet decomposition whereas a matrix input
  undergoes 2D wavelet decomposition.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `y::AbstractArray{T,2} where T<:Number` or `y::AbstractArray{T,3} where T<:Number`:
  Decomposed signal. For an input vector, output is a 2D matrix where each column
  corresponds to a level of decomposition. For an input matrix, output is a 3D array where
  each slice of dimension 3 corresponds to a level of decomposition.

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
xw = Array{eltype(x)}(undef, (8,3))
wt = wavelet(WT.haar)
wpd!(xw, x, wt)

# 2D wavelet decomposition
x = randn(8,8)
xw = Array{eltype(x)}(undef, (8,8,3))
wt = wavelet(WT.haar)
wpd!(xw, x, wt)
```

**See also:** [`wpd`](@ref)
"""
function wpd!(y::AbstractArray{T,2}, 
              x::AbstractArray{T,1},
              wt::DiscreteWavelet, 
              L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    n = length(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)
    @assert size(y) == (n,L+1)

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    y[:,1] = x

    # Wavelet Decomposition
    for i in 0:(L-1)
        np = nodelength(n, i)                   # Parent node length
        for j in 0:((1<<i)-1)
            colₚ = i+1                          # Parent column
            rngₚ = (j*nₚ+1):((j+1)*np)           # Parent range
            @inbounds v = @view y[rngₚ, colₚ]    # Parent node
            colᵣ = colₚ+1                       # Child column
            nc = np÷2                           # Child node length
            rng₁ = (2*j*nc+1):((2*j+1)*nc)      # Approx coef range
            rng₂ = ((2*j+1)*nc+1):(2*(j+1)*nc)  # Detail coef range
            @inbounds w₁ = @view y[rng₁, colᵣ]  # Approx coef node
            @inbounds w₂ = @view y[rng₂, colᵣ]  # Detail coef node
            dwt_step!(w₁, w₂, v, h, g)          # Decompose
        end
    end
    return y
end

# 2D Wavelet Packet Decomposition with allocated output array
function wpd!(y::AbstractArray{T,3},
              x::AbstractArray{T,2},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(x);
              standard::Bool = true) where T<:Number
    # Sanity check
    m, n = size(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)
    @assert size(y) == (m, n,L+1)

    # ----- Allocations and setup to match Wavelets.jl's function requirements -----
    fw = true                                               # forward transform
    si = Vector{T}(undef, length(wt)-1)                     # temp filter vector
    scfilter, dcfilter = WT.makereverseqmfpair(wt, fw, T)   # low & high pass filters
    # First slice of y is level 0, ie. original signal
    y[:,:,1] = x
    # ----- Compute L levels of decomposition -----
    for i in 0:(L-1)
        # Parent node width and height
        mₚ = nodelength(m, i)
        nₚ = nodelength(n, i)
        # Iterate over each nodes at current level
        lrange = 0:((1<<i)-1)
        for j in lrange
            for k in lrange
                # Extract parent node
                sliceₚ = i+1
                rng_row = (j*mₚ+1):((j+1)*mₚ)
                rng_col = (k*nₚ+1):((k+1)*nₚ)
                @inbounds nodeₚ = @view y[rng_row, rng_col, sliceₚ]
                # Extract children nodes of current parent
                sliceᵣ = sliceₚ+1
                @inbounds nodeᵣ = @view y[rng_row, rng_col, sliceᵣ]
                # Perform 1 level of wavelet decomposition
                dwt_step!(nodeᵣ, nodeₚ, wt, dcfilter, scfilter, si, standard=standard)
            end
        end
    end
    return y
end

# ========== Wavelet Packet Transform ==========
# TODO: Add `isvalidquadtree` function in Utils
# 2D Wavelet Packet Transform without allocated output array
"""
    wpt(x, wt[, L; standard])
    wpt(x, wt, tree[; standard])

Returns the wavelet packet transform (WPT) for `L` levels or by given quadratic `tree`.

!!! note
    Nonstandard transform is not yet available. Setting `standard = true` will currently
    result in an error message.

# Arguments
- `x::AbstractVector{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet decomposition whereas a matrix input
  undergoes 2D wavelet decomposition.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Quadtree to transform to be computed accordingly.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,1}` or `::Array{T,2}`: Transformed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
wt = wavelet(WT.haar)
xw = wpt(x, wt)

# 2D wavelet decomposition
x = randn(8,8)
wt = wavelet(WT.haar)
xw = wpt(x, wt)
```

**See also:** [`wpt!`](@ref), [`makequadtree`](@ref)
"""
function Wavelets.Transforms.wpt(x::AbstractArray{T,2}, 
                                 wt::OrthoFilter, 
                                 L::Integer = maxtransformlevels(x);
                                 standard::Bool = true) where T<:Number
    return wpt(x, wt, makequadtree(x, L, :full), standard=standard)
end

function Wavelets.Transforms.wpt(x::AbstractArray{T,2},
                                 wt::OrthoFilter,
                                 tree::BitVector;
                                 standard::Bool = true) where T<:Number
    y = Array{T}(undef, size(x))
    return wpt!(y, x, wt, tree, standard=standard)
end

# 2D Wavelet Packet Transform with allocated output array
"""
    wpt!(y, x, wt[, L; standard])
    wpt!(y, x, wt, tree[; standard])

Same as `wpt` but without array allocation.

!!! note
    Nonstandard transform is not yet available. Setting `standard = true` will currently
    result in an error message.

# Arguments
- `y::AbstractVector{T} where T<:Number` or `y::AbstractArray{T,2} where T<:Number`:
  Allocated output vector/matrix.
- `x::AbstractVector{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet packet transform whereas a matrix input
  undergoes 2D wavelet packet transform.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Quadtree to transform to be computed accordingly.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,1}` or `::Array{T,2}`: Transformed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
xw = similar(x)
wt = wavelet(WT.haar)
wpt!(xw, x, wt)

# 2D wavelet decomposition
x = randn(8,8)
xw = similar(x)
wt = wavelet(WT.haar)
wpt!(xw, x, wt)
```

**See also:** [`wpt`](@ref), [`makequadtree`](@ref)
"""
function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  L::Integer = maxtransformlevels(x),
                                  standard::Bool = true) where T<:Number
    return wpt!(y, x, wt, makequadtree(x, L, :full), standard=standard)
end

function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  tree::BitVector;
                                  standard::Bool = true) where T<:Number
    # Sanity check
    # TODO: define isvalidquadtree
    # @assert isvalidquadtree(x, tree)
    @assert size(y) == size(x)

    # ----- Allocation and setup to match Wavelets.jl's function requirements -----
    m, n = size(x)
    fw = true
    si = Vector{T}(undef, length(wt)-1)                     # temp filter vector
    scfilter, dcfilter = WT.makereverseqmfpair(wt, fw, T)   # low & high pass filters
    temp = copy(x)                                          # temp array
    # ----- Compute transforms based on tree -----
    for i in eachindex(tree)
        # Decompose if node i has children
        if tree[i]
            # Extract parent node
            rng_row = getrowrange(m, i)
            rng_col = getcolrange(n, i)
            @inbounds nodeₚ = @view temp[rng_row, rng_col]
            # Extract children nodes of current parent
            @inbounds nodeᵣ = @view y[rng_row, rng_col]
            # Perform 1 level of wavelet decomposition
            dwt_step!(nodeᵣ, nodeₚ, wt, dcfilter, scfilter, si, standard=standard)
            # If current range not at final iteration, copy to temp
            (4*i<length(tree)) && (temp[rng_row, rng_col] = y[rng_row, rng_col])
        else
            continue
        end
    end
    return y
end

# ========== Inverse Wavelet Packet Transform ==========
# 2D Inverse Wavelet Packet Transform without allocated output array
"""
    wpt(xw, wt[, L; standard])
    wpt(xw, wt, tree[; standard])

Returns the wavelet packet transform (WPT) for `L` levels or by given quadratic `tree`.

!!! note
    Nonstandard transform is not yet available. Setting `standard = true` will currently
    result in an error message.

# Arguments
- `xw::AbstractVector{T} where T<:Number` or `xw::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet reconstruction whereas a matrix input
  undergoes 2D wavelet reconstruction.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Quadtree to transform to be computed accordingly.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,1}` or `::Array{T,2}`: Transformed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
wt = wavelet(WT.haar)
xw = wpt(x, wt)

# 1D wavelet reconstruction
y = iwpt(xw, wt)

# 2D wavelet decomposition
x = randn(8,8)
wt = wavelet(WT.haar)
xw = wpt(x, wt)

# 2D wavelet reconstruction
y = iwpt(xw, wt)
```

**See also:** [`iwpt!`](@ref), [`wpt!`](@ref), [`makequadtree`](@ref)
"""
function Wavelets.Transforms.iwpt(xw::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  L::Integer = maxtransformlevels(xw);
                                  standard::Bool = true) where T<:Number
    return iwpt(xw, wt, makequadtree(xw, L, :full), standard=standard)
end

function Wavelets.Transforms.iwpt(xw::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  tree::BitVector;
                                  standard::Bool = true) where T<:Number
    x̂ = similar(xw)
    return iwpt!(x̂, xw, wt, tree, standard=standard)
end

# 2D Inverse Wavelet Packet Transform with allocated output array
"""
    iwpt!(x̂, xw, wt[, L; standard])
    iwpt!(x̂, xw, wt, tree[; standard])

Same as `wpt` but without array allocation.

!!! note
    Nonstandard transform is not yet available. Setting `standard = true` will currently
    result in an error message.

# Arguments
- `x̂::AbstractVector{T} where T<:Number` or `x̂::AbstractArray{T,2} where T<:Number`:
  Allocated output vector/matrix.
- `xw::AbstractVector{T} where T<:Number` or `xw::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet packet reconstruction whereas a matrix
  input undergoes 2D wavelet packet reconstruction.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Quadtree to transform to be computed accordingly.
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,1}` or `::Array{T,2}`: Reconstructed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
xw = similar(x)
wt = wavelet(WT.haar)
wpt!(xw, x, wt)

# 1D wavelet reconstruction
y = similar(x)
iwpt!(y, xw, wt)

# 2D wavelet decomposition
x = randn(8,8)
xw = similar(x)
wt = wavelet(WT.haar)
wpt!(xw, x, wt)

# 2D wavelet reconstruction
y = similar(x)
iwpt!(y, xw, wt)
```

**See also:** [`iwpt`](@ref), [`wpt!`](@ref) [`makequadtree`](@ref)
"""
function Wavelets.Transforms.iwpt!(x̂::AbstractArray{T,2},
                                   xw::AbstractArray{T,2},
                                   wt::OrthoFilter,
                                   L::Integer = maxtransformlevels(xw);
                                   standard::Bool = true) where T<:Number
    return iwpt!(x̂, xw, wt, makequadtree(xw, L, :full), standard=standard)
end

function Wavelets.Transforms.iwpt!(x̂::AbstractArray{T,2},
                                   xw::AbstractArray{T,2},
                                   wt::OrthoFilter,
                                   tree::BitVector;
                                   standard::Bool = true) where T<:Number
    # Sanity check
    # TODO: define isvalidquadtree
    # @assert isvalidquadtree(xw, tree)
    @assert size(x̂) == size(xw)

    # ----- Allocation and setup to match Wavelets.jl's function requirements -----
    m, n = size(xw)
    fw = false
    si = Vector{T}(undef, length(wt)-1)                     # temp filter vector
    scfilter, dcfilter = WT.makereverseqmfpair(wt, fw, T)   # low & high pass filters
    temp = copy(xw)                                         # temp array
    # ----- Compute transforms based on tree -----
    for i in reverse(eachindex(tree))
        # Reconstruct to node i if it has children
        if tree[i]
            # Extract parent node
            rng_row = getrowrange(m, i)
            rng_col = getcolrange(n, i)
            @inbounds nodeₚ = @view x̂[rng_row, rng_col]         # Parent node
            @inbounds nodeᵣ = @view temp[rng_row, rng_col]      # Children nodes
            # Perform 1 level of wavelet reconstruction
            idwt_step!(nodeₚ, nodeᵣ, wt, dcfilter, scfilter, si, standard=standard)
            # If current range not at final iteration, copy to temp
            (i>1) && (temp[rng_row, rng_col] = x̂[rng_row, rng_col])
        else
            continue
        end
    end
    return x̂
end

include("wt_one_level.jl")
include("wt_all.jl")

end # end module