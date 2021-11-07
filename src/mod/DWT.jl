module DWT
export 
    # WPD
    wpd,
    wpd!,
    iwpd,
    iwpd!,
    # Transform on all signals
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
    wpd(x, wt[, L; standard])

Returns the wavelet packet decomposition (WPD) for L levels for input signal x.

!!! note 
    Nonstandard transform is not yet available for 2D-WPD. Setting `standard = true`
    will currently result in an error message.

# Arguments
- `x::AbstractVector{T} where T<:Number` or `x::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet decomposition whereas a matrix input
  undergoes 2D wavelet decomposition.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,2}` or `::Array{T,3}`: Decomposed signal. For an input vector `x`, output is a
  2D matrix where each column corresponds to a level of decomposition. For an input matrix
  `x`, output is a 3D array where each slice of dimension 3 corresponds to a level of
  decomposition.

# Examples:
```julia
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

**See also:** [`wpd!`](@ref), [`iwpd`](@ref)
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

# Keyword Arguments
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

**See also:** [`wpd`](@ref), [`iwpd`](@ref)
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
        nₚ = nodelength(n, i)                   # Parent node length
        for j in 0:((1<<i)-1)
            colₚ = i+1                          # Parent column
            rngₚ = (j*nₚ+1):((j+1)*nₚ)           # Parent range
            @inbounds v = @view y[rngₚ, colₚ]    # Parent node
            colᵣ = colₚ+1                       # Child column
            nᵣ = nₚ÷2                           # Child node length
            rng₁ = (2*j*nᵣ+1):((2*j+1)*nᵣ)      # Approx coef range
            rng₂ = ((2*j+1)*nᵣ+1):(2*(j+1)*nᵣ)  # Detail coef range
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

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)   # low & high pass filters
    temp = Array{T,2}(undef, (m,n))

    # First slice of y is level 0, ie. original signal
    @inbounds y[:,:,1] = x
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
                @inbounds v = @view y[rng_row, rng_col, sliceₚ]
                # Extract children nodes of current parent
                sliceᵣ = sliceₚ+1
                mᵣ = mₚ÷2
                nᵣ = nₚ÷2
                @inbounds w₁ = @view y[(2*j*mᵣ+1):((2*j+1)*mᵣ), (2*k*nᵣ+1):((2*k+1)*nᵣ), sliceᵣ]
                @inbounds w₂ = @view y[(2*j*mᵣ+1):((2*j+1)*mᵣ), ((2*k+1)*nᵣ+1):(2*(k+1)*nᵣ), sliceᵣ]
                @inbounds w₃ = @view y[((2*j+1)*mᵣ+1):(2*(j+1)*mᵣ), (2*k*nᵣ+1):((2*k+1)*nᵣ), sliceᵣ]
                @inbounds w₄ = @view y[((2*j+1)*mᵣ+1):(2*(j+1)*mᵣ), ((2*k+1)*nᵣ+1):(2*(k+1)*nᵣ), sliceᵣ]
                # Extract temp subarray (Same size as parent node)
                @inbounds tempₖ = @view temp[rng_row, rng_col]
                # Perform 1 level of wavelet decomposition
                @inbounds dwt_step!(w₁, w₂, w₃, w₄, v, h, g, tempₖ, standard=standard)
            end
        end
    end
    return y
end

# ========== Inverse Wavelet Packet Decomposition ==========
# IWPD by level without allocation
"""
    iwpd(xw, wt[, L; standard])
    iwpd(xw, wt, tree[; standard])

Computes the inverse wavelet packet decomposition (IWPD) for `L` levels or by given `tree.`

!!! note 
    Nonstandard transform is not yet available. Setting `standard = true` will
    currently result in an error message.

# Arguments
- `x̂::AbstractVector{T} where T<:Number` or `x̂::AbstractArray{T,2} where T<:Number`:
  Allocated output vector/matrix for reconstructed signal.
- `xw::AbstractArray{T,2} where T<:Number` or `xw::AbstractArray{T,3} where T<:Number`:
  Input array. A 2D input undergoes 1D wavelet reconstruction whereas a 3D input undergoes
  2D wavelet reconstruction.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels of wavelet
  decomposition.
- `tree::BitVector`: Binary tree or Quadtree for inverse transform to be computed
  accordingly.

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `x̂::Array{T,1}` or `x̂::Array{T,2}`: Reconstructed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)

# 1D wavelet reconstruction
y = iwpd(xw, wt)

# 2D wavelet decomposition
x = randn(8,8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)

# 2D wavelet reconstruction
y = iwpd(xw, wt)
```

**See also:** [`iwpd!`](@ref), [`iwpt`](@ref)
"""
function iwpd(xw::AbstractArray{T},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(xw,1);
              kwargs...) where T<:Number
    @assert ndims(xw) ≥ 2
    dim = size(xw)[1:(end-1)]
    x̂ = Array{T}(undef, dim)
    iwpd!(x̂, xw, wt, L, kwargs...)
    return x̂
end

# IWPD by tree without allocation
function iwpd(xw::AbstractArray{T}, 
              wt::OrthoFilter, 
              tree::BitVector; 
              kwargs...) where T<:Number
    @assert ndims(xw) ≥ 2
    dim = size(xw)[1:(end-1)]
    x̂ = Array{T}(undef, dim)
    iwpd!(x̂, xw, wt, tree, kwargs...)
    return x̂
end

# 1D IWPD by level with allocation
"""
    iwpd(x̂, xw, wt[, L; standard])
    iwpd(x̂, xw, wt, tree[; standard])

Same as `iwpd` but without array allocation.

!!! note 
    Nonstandard transform is not yet available. Setting `standard = true` will
    currently result in an error message.

# Arguments
- `x̂::AbstractVector{T} where T<:Number` or `x̂::AbstractArray{T,2} where T<:Number`:
  Allocated output vector/matrix for reconstructed signal.
- `xw::AbstractArray{T,2} where T<:Number` or `xw::AbstractArray{T,3} where T<:Number`:
  Input array. A 2D input undergoes 1D wavelet reconstruction whereas a 3D input undergoes
  2D wavelet reconstruction.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Binary tree or Quadtree to transform to be computed accordingly.

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `x̂::Array{T,1}` or `x̂::Array{T,2}`: Reconstructed signal. 

# Examples:
```
using Wavelets, WaveletsExt

# 1D wavelet decomposition
x = randn(8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)
y = similar(x)

# 1D wavelet reconstruction
iwpd!(y, xw, wt)

# 2D wavelet decomposition
x = randn(8,8)
wt = wavelet(WT.haar)
xw = wpd(x, wt)
y = similar(x)

# 2D wavelet reconstruction
iwpd!(y, xw, wt)
```

**See also:** [`iwpd`](@ref), [`iwpt`](@ref)
"""
function iwpd!(x̂::AbstractVector{T}, 
               xw::AbstractArray{T,2},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x̂)) where T<:Number
    iwpd!(x̂, xw, wt, maketree(length(x̂), L, :full))
    return x̂
end

# 2D IWPD by level with allocation
function iwpd!(x̂::AbstractArray{T,2},
               xw::AbstractArray{T,3},
               wt::OrthoFilter,
               L::Integer = maxtransformlevels(x̂);
               standard::Bool = true) where T<:Number
    iwpd!(x̂, xw, wt, maketree(size(x̂)..., L, :full), standard=standard)
    return x̂
end

# 1D IWPD by tree with allocation
function iwpd!(x̂::AbstractVector{T},
               xw::AbstractArray{T,2},
               wt::OrthoFilter,
               tree::BitVector) where T<:Number
    # Sanity check
    @assert size(x̂,1) == size(xw,1)
    @assert isvalidtree(x̂, tree)
    # Get basis coefficients then compute iwpt
    w = getbasiscoef(xw, tree)
    iwpt!(x̂, w, wt, tree)
    return x̂
end

# 2D IWPD by tree with allocation
function iwpd!(x̂::AbstractArray{T,2},
               xw::AbstractArray{T,3},
               wt::OrthoFilter,
               tree::BitVector;
               standard::Bool = true) where T<:Number
    # Sanity check
    @assert size(x̂,1) == size(xw,1)
    @assert size(x̂,2) == size(xw,2)
    @assert isvalidtree(x̂, tree)

    # Setup
    m, n, _ = size(xw)
    g, h = WT.makereverseqmfpair(wt, true)
    xwₜ = copy(xw)
    temp = Array{T,2}(undef, (m,n))

    # Inverse transform
    for i in reverse(eachindex(tree))
        # Reconstruct node i if it has children
        if tree[i]
            # Extract parent node
            d = getdepth(i,:quad)
            rows = getrowrange(m, i)
            cols = getcolrange(n, i)
            @inbounds v = d == 0 ? x̂ : @view xwₜ[rows, cols, d+1]
            # Extract children nodes of current parent
            dᵣ = d+1
            rows₁ = getrowrange(m, getchildindex(i,:topleft))
            cols₁ = getcolrange(n, getchildindex(i,:topleft))
            rows₂ = getrowrange(m, getchildindex(i,:topright))
            cols₂ = getcolrange(n, getchildindex(i,:topright))
            rows₃ = getrowrange(m, getchildindex(i,:bottomleft))
            cols₃ = getcolrange(n, getchildindex(i,:bottomleft))
            rows₄ = getrowrange(m, getchildindex(i,:bottomright))
            cols₄ = getcolrange(n, getchildindex(i,:bottomright))
            @inbounds w₁ = @view xwₜ[rows₁, cols₁, dᵣ+1]
            @inbounds w₂ = @view xwₜ[rows₂, cols₂, dᵣ+1]
            @inbounds w₃ = @view xwₜ[rows₃, cols₃, dᵣ+1]
            @inbounds w₄ = @view xwₜ[rows₄, cols₄, dᵣ+1]
            # Extract temp subarray (Same size as parent node)
            @inbounds tempᵢ = @view temp[rows, cols]
            # Perform 1 level of wavelet reconstruction
            @inbounds idwt_step!(v, w₁, w₂, w₃, w₄, h, g, tempᵢ, standard=standard)
        else
            continue
        end
    end
    return x̂
end

# ========== Wavelet Packet Transform ==========
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

# Keyword Arguments
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

**See also:** [`wpt!`](@ref), [`maketree`](@ref)
"""
function Wavelets.Transforms.wpt(x::AbstractArray{T,2}, 
                                 wt::OrthoFilter, 
                                 L::Integer = maxtransformlevels(x);
                                 standard::Bool = true) where T<:Number
    return wpt(x, wt, maketree(size(x)..., L, :full), standard=standard)
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

# Keyword Arguments
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

**See also:** [`wpt`](@ref), [`maketree`](@ref)
"""
function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  L::Integer = maxtransformlevels(x),
                                  standard::Bool = true) where T<:Number
    return wpt!(y, x, wt, maketree(size(x)..., L, :full), standard=standard)
end

function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  tree::BitVector;
                                  standard::Bool = true) where T<:Number
    # Sanity check
    @assert isvalidtree(x, tree)
    @assert size(y) == size(x)

    # ----- Allocation and setup to match Wavelets.jl's function requirements -----
    m, n = size(x)
    g, h = WT.makereverseqmfpair(wt, true)      # low & high pass filters
    yₜ = copy(x)                                 # Placeholder of y
    temp = Array{T,2}(undef, (m,n))             # Temp array
    # ----- Compute transforms based on tree -----
    for i in eachindex(tree)
        # Decompose if node i has children
        if tree[i]
            # Extract parent node
            rows = getrowrange(m, i)
            cols = getcolrange(n, i)
            @inbounds v = @view yₜ[rows, cols]
            # Extract children nodes of current parent
            rows₁ = getrowrange(m, getchildindex(i,:topleft))
            cols₁ = getcolrange(n, getchildindex(i,:topleft))
            rows₂ = getrowrange(m, getchildindex(i,:topright))
            cols₂ = getcolrange(n, getchildindex(i,:topright))
            rows₃ = getrowrange(m, getchildindex(i,:bottomleft))
            cols₃ = getcolrange(n, getchildindex(i,:bottomleft))
            rows₄ = getrowrange(m, getchildindex(i,:bottomright))
            cols₄ = getcolrange(n, getchildindex(i,:bottomright))
            @inbounds w₁ = @view y[rows₁, cols₁]
            @inbounds w₂ = @view y[rows₂, cols₂]
            @inbounds w₃ = @view y[rows₃, cols₃]
            @inbounds w₄ = @view y[rows₄, cols₄]
            # Extract temp subarray (Same size as parent node)
            @inbounds tempᵢ = @view temp[rows, cols]
            # Perform 1 level of wavelet decomposition
            @inbounds dwt_step!(w₁, w₂, w₃, w₄, v, h, g, tempᵢ, standard=standard)
            # If current range not at final iteration, copy to yₜ
            if 4*i<length(tree)
                @inbounds yₜ[rows, cols] = @view y[rows, cols]
            end
        else
            continue
        end
    end
    return y
end

# ========== Inverse Wavelet Packet Transform ==========
# 2D Inverse Wavelet Packet Transform without allocated output array
"""
    iwpt(xw, wt[, L; standard])
    iwpt(xw, wt, tree[; standard])

Returns the inverse wavelet packet transform (IWPT) for `L` levels or by given quadratic
`tree`.

!!! note 
    Nonstandard transform is not yet available. Setting `standard = true` will
    currently result in an error message.

# Arguments
- `xw::AbstractVector{T} where T<:Number` or `xw::AbstractArray{T,2} where T<:Number`: Input
  vector/matrix. A vector input undergoes 1D wavelet reconstruction whereas a matrix input
  undergoes 2D wavelet reconstruction.
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of levels for wavelet
  decomposition.
- `tree::BitVector`: Quadtree to transform to be computed accordingly.

# Keyword Arguments
- `standard::Bool`: (Default: `true`) Whether to compute the standard or non-standard
  wavelet transform. Only applicable for 2D signals.

# Returns
- `::Array{T,1}` or `::Array{T,2}`: Reconstructed signal. 

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

**See also:** [`iwpt!`](@ref), [`wpt!`](@ref), [`maketree`](@ref)
"""
function Wavelets.Transforms.iwpt(xw::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  L::Integer = maxtransformlevels(xw);
                                  standard::Bool = true) where T<:Number
    return iwpt(xw, wt, maketree(size(xw)..., L, :full), standard=standard)
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

Same as `iwpt` but without array allocation.

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

**See also:** [`iwpt`](@ref), [`wpt!`](@ref) [`maketree`](@ref)
"""
function Wavelets.Transforms.iwpt!(x̂::AbstractArray{T,2},
                                   xw::AbstractArray{T,2},
                                   wt::OrthoFilter,
                                   L::Integer = maxtransformlevels(xw);
                                   standard::Bool = true) where T<:Number
    return iwpt!(x̂, xw, wt, maketree(size(xw)..., L, :full), standard=standard)
end

function Wavelets.Transforms.iwpt!(x̂::AbstractArray{T,2},
                                   xw::AbstractArray{T,2},
                                   wt::OrthoFilter,
                                   tree::BitVector;
                                   standard::Bool = true) where T<:Number
    # Sanity check
    @assert isvalidtree(xw, tree)
    @assert size(x̂) == size(xw)

    # ----- Setup -----
    m, n = size(xw)
    g, h = WT.makereverseqmfpair(wt, true)      # low & high pass filters
    xwₜ = copy(xw)                               # Placeholder for xw
    temp = Array{T,2}(undef, (m,n))             # temp array
    # ----- Compute transforms based on tree -----
    for i in reverse(eachindex(tree))
        # Reconstruct to node i if it has children
        if tree[i]
            # Extract parent node
            rows = getrowrange(m, i)
            cols = getcolrange(n, i)
            @inbounds v = @view x̂[rows, cols]
            # Extract children nodes of current parent
            rows₁ = getrowrange(m, getchildindex(i,:topleft))
            cols₁ = getcolrange(n, getchildindex(i,:topleft))
            rows₂ = getrowrange(m, getchildindex(i,:topright))
            cols₂ = getcolrange(n, getchildindex(i,:topright))
            rows₃ = getrowrange(m, getchildindex(i,:bottomleft))
            cols₃ = getcolrange(n, getchildindex(i,:bottomleft))
            rows₄ = getrowrange(m, getchildindex(i,:bottomright))
            cols₄ = getcolrange(n, getchildindex(i,:bottomright))
            @inbounds w₁ = @view xwₜ[rows₁, cols₁]
            @inbounds w₂ = @view xwₜ[rows₂, cols₂]
            @inbounds w₃ = @view xwₜ[rows₃, cols₃]
            @inbounds w₄ = @view xwₜ[rows₄, cols₄]
            # Extract temp subarray (Same size as parent node)
            @inbounds tempᵢ = @view temp[rows, cols]
            # Perform 1 level of wavelet reconstruction
            @inbounds idwt_step!(v, w₁, w₂, w₃, w₄, h, g, tempᵢ, standard=standard)
            # If current range not at final iteration, copy to temp
            if i > 1
                @inbounds xwₜ[rows, cols] = v
            end
        else
            continue
        end
    end
    return x̂
end

include("dwt_one_level.jl")
include("dwt_all.jl")

end # end module