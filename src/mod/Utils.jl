module Utils
export 
    makequadtree,
    getquadtreelevel,
    getrowrange,
    getcolrange,
    left,
    right,
    nodelength,
    getleaf,
    coarsestscalingrange,
    finestdetailrange,
    relativenorm,
    psnr,
    snr,
    ssim,
    duplicatesignals,
    generatesignals,
    ClassData,
    generateclassdata

using
    Wavelets,
    LinearAlgebra,
    Distributions,
    Random, 
    ImageQualityIndexes

# ========== Structs ==========
"""
    ClassData(type, s₁, s₂, s₃)

Based on the input `type`, generates 3 classes of signals with sample sizes
`s₁`, `s₂`, and `s₃` respectively. Accepted input types are:  
- `:tri`: Triangular signals of length 32
- `:cbf`: Cylinder-Bell-Funnel signals of length 128

Based on N. Saito and R. Coifman in "Local Discriminant Basis and their Applications" in the
Journal of Mathematical Imaging and Vision, Vol. 5, 337-358 (1995).

**See also:** [`generateclassdata`](@ref)
"""
struct ClassData
    "Signal type, accepted inputs are `:tri` and `:cbf`"
    type::Symbol
    "Sample size for class 1"
    s₁::Int
    "Sample size for class 2"
    s₂::Int
    "Sample size for class 3"
    s₃::Int
    ClassData(type, s₁, s₂, s₃) = type ∈ [:tri, :cbf] ? new(type, s₁, s₂, s₃) : 
        throw(ArgumentError("Invalid type. Accepted types are :tri and :cbf only."))
end

# ========== Functions ==========
# `maxtransformlevels` of a specific dimension
"""
    maxtransformlevels(n)

    maxtransformlevels(x[, dims])

Extension function from Wavelets.jl. Finds the max number of transform levels for an array
`x`. If `dims` is provided for a multidimensional array, then it finds the max transform
level for that corresponding dimension.

# Arguments
- `x::AbstractArray`: Input array.
- `dims::Integer`: Dimension used for wavelet transform.
- `n::Integer`: Length of input vector.

# Returns
`::Integer`: Max number of transform levels.

# Examples
```@repl
using Wavelets, WaveletsExt

# Define random signal
x = randn(64, 128)

# Max transform levels and their corresponding return values
maxtransformlevels(128)     # 7
maxtransformlevels(x)       # 6
maxtransformlevels(x, 2)    # 7
```
"""
function Wavelets.Util.maxtransformlevels(x::AbstractArray, dims::Integer)
    # Sanity check
    @assert 1 ≤ dims ≤ ndims(x)
    # Compute max transform levels of relevant dimension
    return maxtransformlevels(size(x)[dims])
end

# Left and right node index of a binary tree
"""
    left(i)

Given the node index `i`, returns the index of its left node.

# Arguments
- `i::Integer`: Index of the node of interest.

# Returns
`::Integer`: Index of left child.

# Examples
```@repl
using Wavelets, WaveletsExt

left(3)     # 6
```

**See also:** [`right`](@ref)
"""
left(i::Integer) = i<<1

"""
    right(i)

Given the node index `i`, returns the index of its right node.

# Arguments
- `i::Integer`: Index of the node of interest.

# Returns
`::Integer`: Index of right child.

# Examples
```@repl
using Wavelets, WaveletsExt

left(3)     # 7
```

**See also:** [`left`](@ref)
"""
right(i::Integer) = i<<1 + 1

# Length of node at level L
"""
    nodelength(N, L)

Returns the node length at level L of a signal of length N when performaing wavelet packet
decomposition. Level L == 0 corresponds to the original input signal.

# Arguments
- `N::Integer`: Length of signal.
- `L::Integer`: Level of signal.

# Returns
`::Integer`: Length of nodes at level `L`.
"""
nodelength(N::Integer, L::Integer) = N >> L

# Get leaf nodes in the form of a BitVector
"""
    getleaf(tree)

Returns the leaf nodes of a tree.

# Arguments
- `tree::BitVector`: BitVector to represent binary tree.

# Returns
`::BitVector`: BitVector that can represent a binary tree, but only the leaves are labeled
1, the rest of the nodes are labeled 0.

# Examples
```@repl
using Wavelets, WaveletsExt

tree = maketree(4, 2, :dwt)     # [1,1,0]
getleaf(tree)                   # [0,0,1,1,1,0,0]
```
"""
function getleaf(tree::BitVector)
    @assert isdyadic(length(tree) + 1)

    result = falses(2*length(tree) + 1)
    result[1] = 1
    for i in 1:length(tree)
        if tree[i] == 0
            continue
        else
            result[i] = 0
            result[left(i)] = 1
            result[right(i)] = 1
        end
    end
    
    return result
end

# Packet table Indexing
"""
    packet(d, b, n)

Packet table indexing.

# Arguments
- `d::Integer`: Depth of splitting in packet decomposition. *Note: Depth of root node is 0.*
- `b::Integer`: Block index among 2ᵈ possibilities at depth `d`. *Note: Block indexing
  starts from 0.*
- `n::Integer`: Length of signal.

# Returns
`::UnitRange{Int64}`: Index range of block `b` of signal of length `n` at level `d`.

# Examples
```julia
using WaveletsExt

WaveletsExt.Utils.packet(0, 0, 8)       # 1:8
WaveletsExt.Utils.packet(2, 1, 8)       # 3:4
```

Translated from Wavelab850 by Zeng Fung Liew.
"""
function packet(d::Integer, b::Integer, n::Integer)
    npack = 1 << d                                  # Number of blocks in current level
    p = (b * (n÷npack) + 1) : ((b+1) * (n÷npack))   # Range of specified block
    return p
end

# Index range of coarsest scaling coefficients
"""
    coarsestscalingrange(x, tree[, redundant])
    coarsestscalingrange(n, tree[, redundant])

Given a binary tree, returns the index range of the coarsest scaling coefficients.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Decomposed 1D-signal.
- `n::Integer`: Length of signal of interest.
- `tree::BitVector`: Binary tree.
- `redundant::Bool`: (Default: `false`) Whether the wavelet decomposition is redundant.
  Examples of redundant wavelet transforms are the Autocorrelation wavelet transform (ACWT),
  Stationary wavelet transform (SWT), and the Maximal Overlap wavelet transform (MOWT).

# Returns
`UnitRange{Integer}` or `::Tuple{UnitRange{Integer}, Integer}`: The index range of the
coarsest scaling subspace based on the input binary tree.

# Examples
```@repl
using Wavelets, WaveletsExt

x = randn(8)
wt = wavelet(WT.haar)
tree = maketree(x)

# Non-redundant wavelet transform
xw = wpd(x, wt)
coarsestscalingrange(xw, tree)          # 1:1

# Redundant wavelet transform
xw = swpd(x, wt)
coarsestscalingrange(xw, tree, true)    # (1:8, 8)
```

*See also:* [`finestdetailrange`](@ref)
"""
function coarsestscalingrange(x::AbstractArray{T}, 
                              tree::BitVector, 
                              redundant::Bool=false) where T<:Number
    return coarsestscalingrange(size(x,1), tree, redundant)
end

function coarsestscalingrange(n::Integer, tree::BitVector, redundant::Bool=false)
    if !redundant          # regular wt
        i = 1
        j = 0
        while i<length(tree) && tree[i]       # has children
            i = left(i)
            j += 1
        end
        rng = 1:(n>>j) 
    else                   # redundant wt
        i = 1
        while i<length(tree) && tree[i]       # has children
            i = left(i)
        end
        rng = (1:n, i)
    end
    return rng
end

# Index range of finest detail coefficients
"""
    finestdetailrange(x, tree[, redundant])
    finestdetailrange(n, tree[, redundant])

Given a binary tree, returns the index range of the finest detail coefficients.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Decomposed 1D-signal.
- `n::Integer`: Length of signal of interest.
- `tree::BitVector`: Binary tree.
- `redundant::Bool`: (Default: `false`) Whether the wavelet decomposition is redundant.
  Examples of redundant wavelet transforms are the Autocorrelation wavelet transform (ACWT),
  Stationary wavelet transform (SWT), and the Maximal Overlap wavelet transform (MOWT).

# Returns
`UnitRange{Integer}` or `::Tuple{UnitRange{Integer}, Integer}`: The index range of the
finest detail subspace based on the input binary tree.

# Examples
```@repl
using Wavelets, WaveletsExt

x = randn(8)
wt = wavelet(WT.haar)
tree = maketree(x)

# Non-redundant wavelet transform
xw = wpd(x, wt)
finestdetailrange(xw, tree)          # 8:8

# Redundant wavelet transform
xw = swpd(x, wt)
finestdetailrange(xw, tree, true)    # (1:8, 15)
```

*See also:* [`coarsestscalingrange`](@ref)
"""
function finestdetailrange(x::AbstractArray{T}, tree::BitVector,
        redundant::Bool=false) where T<:Number

    return finestdetailrange(size(x,1), tree, redundant)
end

function finestdetailrange(n::Integer, tree::BitVector, redundant::Bool=false)
    if !redundant      # regular wt
        i = 1
        j = 0
        while i≤length(tree) && tree[i]
            i = right(i)
            j += 1
        end
        n₀ = nodelength(n, j)
        rng = (n-n₀+1):n
    else               # redundant wt
        i = 1
        while i≤length(tree) && tree[i]
            i = right(i)
        end
        rng = (1:n, i)
    end
    return rng
end

# Build a quadtree
"""
    makequadtree(x, L[, s])

Build quadtree for 2D wavelet transform. Indexing of the quadtree are as follows:

```
Level 0                 Level 1                 Level 2             ...
_________________       _________________       _________________
|               |       |   2   |   3   |       |_6_|_7_|10_|11_|
|       1       |       |_______|_______|       |_8_|_9_|12_|13_|   ...
|               |       |   4   |   5   |       |14_|15_|18_|19_|
|_______________|       |_______|_______|       |16_|17_|20_|21_|
```

# Arguments
- `x::AbstractArray{T,2} where T<:Number`: Input array.
- `L::Integer`: Number of decomposition levels.
- `s::Symbol`: (Default: `:full`) Type of quadtree. Available types are `:full` and `:dwt`.

# Returns
`::BitVector`: Quadtree representation.

# Examples
```@repl
using WaveletsExt

x = randn(16,16)
makequadtree(x, 3)
```
"""
function makequadtree(x::AbstractArray{T,2}, L::Integer, s::Symbol = :full) where T<:Number
    ns = maxtransformlevels(x)      # Max transform levels of x
    # TODO: Find a mathematical formula to define this rather than sum things up to speed up
    # TODO: process.
    nq = sum(4 .^ (0:ns))           # Quadtree size
    @assert 0 ≤ L ≤ ns

    # Construct quadtree
    q = BitArray(undef, nq)
    fill!(q, false)

    # Fill in true values depending on input `s`
    if s == :full
        # TODO: Find a mathematical formula to define this
        rng = 4 .^ (0:(L-1)) |> sum |> x -> 1:x
        for i in rng
            @inbounds q[i] = true
        end
    elseif s == :dwt
        q[1] = true     # Root node
        for i in 0:(L-2)
            # TODO: Find a mathematical formula to define this
            idx = 4 .^ (0:i) |> sum |> x -> x+1     # Subsequent LL subspace nodes
            @inbounds q[idx] = true
        end
    else
        throw(ArgumentError("Unknown symbol."))
    end
    return q
end

# Get level of a particular index in a tree
"""
    getquadtreelevel(idx)

Get level of `idx` in the quadtree.

# Arguments
- `idx::T where T<:Integer`: Index of a quadtree.

# Returns
`::T`: Level of `idx`.

# Examples
```@repl
using WaveletsExt

getquadtreelevel(1)     # 0
getquadtreelevel(3)     # 1
```

**See also:** [`makequadtree`](@ref)
"""
function getquadtreelevel(idx::T) where T<:Integer
    # TODO: Get a mathematical solution to this
    @assert idx > 0
    # Root node => level 0
    if idx == 1
        return 0
    # Node level is 1 level lower than parent node
    else
        parent_idx = floor(T, (idx+2)/4)
        return 1 + getquadtreelevel(parent_idx)
    end
end

# TODO: Build a generalizable function `getslicerange` that does the same thing as
# TODO: `getrowrange` and `getcolrange` but condensed in to 1 function and generalizes into
# TODO: any arbitrary number of dimensions.
# Get range of particular row from a given signal length and quadtree index
"""
    getrowrange(n, idx)

Get the row range from a matrix with `n` rows that corresponds to `idx` from a quadtree.

# Arguments
- `n::Integer`: Number of rows in matrix.
- `idx::T where T<:Integer`: Index from a quadtree corresponding to the matrix.

# Returns
`::UnitRange{Int64}`: Row range in matrix that corresponds to `idx`.

# Examples
```@repl
using WaveletsExt

x = randn(8,8)
tree = makequadtree(x, 3, :full)
getrowrange(8,3)            # 1:4
```

**See also:** [`makequadtree`](@ref), [`getcolrange`](@ref)
"""
function getrowrange(n::Integer, idx::T) where T<:Integer
    # TODO: Get a mathematical solution to this, if possible
    # Sanity check
    @assert idx > 0

    # Root node => range is of entire array
    if idx == 1
        return 1:n
    # Slice from parent node
    else
        # Get parent node's row range and midpoint of the range.
        parent_idx = floor(T, (idx+2)/4)
        parent_rng = getrowrange(n, parent_idx)
        midpoint = (parent_rng[begin]+parent_rng[end]) ÷ 2
        # Children nodes of subspaces LL and HL have indices 4i-2 and 4i-1 for any parent of
        # index i, these nodes take the upper half of the parent's range.
        if idx < 4*parent_idx
            return parent_rng[begin]:midpoint
        # Children nodes of subspaces LH and HH have indices 4i and 4i+1 for any parent of
        # index i, these nodes take the lower half of the parent's range.
        else
            return (midpoint+1):parent_rng[end]
        end
    end
end

# Get range of particular column from a given signal length and quadtree index
"""
    getcolrange(n, idx)

Get the column range from a matrix with `n` columns that corresponds to `idx` from a
quadtree.

# Arguments
- `n::Integer`: Number of columns in matrix.
- `idx::T where T<:Integer`: Index from a quadtree corresponding to the matrix.

# Returns
`::UnitRange{Int64}`: Column range in matrix that corresponds to `idx`.

# Examples
```@repl
using WaveletsExt

x = randn(8,8)
tree = makequadtree(x, 3, :full)
getcolrange(8,3)            # 5:8
```

**See also:** [`makequadtree`](@ref), [`getrowrange`](@ref)
"""
function getcolrange(n::Integer, idx::T) where T<:Integer
    # TODO: Get a mathematical solution to this, if possible
    # Sanity check
    @assert idx > 0

    # Root node => range is of entire array
    if idx == 1
        return 1:n
    # Slice from parent node
    else
        # Get parent node's column range and midpoint of the range.
        parent_idx = floor(T, (idx+2)/4)
        parent_rng = getcolrange(n, parent_idx)
        midpoint = (parent_rng[begin]+parent_rng[end]) ÷ 2
        # Children nodes of subspaces LL and LH have indices 4i-2 and 4i for any parent of
        # index i, these nodes take the left half of the parent's range.
        if iseven(idx)
            return parent_rng[begin]:midpoint
        # Children nodes of subspaces HL and HH have indices 4i-1 and 4i+1 for any parent of
        # index i, these nodes take the right half of the parent's range.
        else
            return (midpoint+1):parent_rng[end]
        end
    end
end

# ----- Computational metrics -----
# Relative norm between 2 vectors
"""
    relativenorm(x, x₀[, p]) where T<:Number

Returns the relative norm of base p between original signal x₀ and noisy signal
x.

# Arguments
- `x::AbstractVector{T} where T<:Number`: Signal with noise.
- `x₀::AbstractVector{T} where T<:Number`: Reference signal.
- `p::Real`: (Default: 2) ``p``-norm to be computed.

# Returns
`::AbstractFloat`: The relative norm between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

relativenorm(x, y)
```

**See also:** [`psnr`](@ref), [`snr`](@ref), [`ssim`](@ref)
"""
function relativenorm(x::AbstractVector{T}, 
                      x₀::AbstractVector{T}, 
                      p::Real = 2) where T<:Number
    @assert length(x) == length(x₀)             # ensure same lengths
    return norm(x-x₀,p)/norm(x₀,p)
end

# PSNR between 2 vectors
@doc raw"""
    psnr(x, x₀)

Returns the peak signal to noise ratio (PSNR) between original signal x₀ and noisy signal x.

PSNR definition: ``10 \log_{10} \frac{\max{(x_0)}^2}{MSE(x, x_0)}``

# Arguments
- `x::AbstractVector{T} where T<:Number`: Signal with noise.
- `x₀::AbstractVector{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The PSNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

psnr(x, y)
```

**See also:** [`relativenorm`](@ref), [`snr`](@ref), [`ssim`](@ref)
"""
function psnr(x::AbstractVector{T}, x₀::AbstractVector{T}) where T<:Number
    @assert length(x) == length(x₀)              # ensure same lengths
    sse = zero(T)
    for i in eachindex(x)
        @inbounds sse += (x[i] - x₀[i])^2
    end
    mse = sse/length(x)
    return 20 * log(10, maximum(x₀)) - 10 * log(10, mse)
end

# SNR between 2 vectors
@doc raw"""
    snr(x, x₀)

Returns the signal to noise ratio (SNR) between original signal x₀ and noisy signal x.

SNR definition: ``20 \log_{10} \frac{||x_0||_2}{||x-x_0||_2}``

# Arguments
- `x::AbstractVector{T} where T<:Number`: Signal with noise.
- `x₀::AbstractVector{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The SNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

snr(x, y)
```

**See also:** [`relativenorm`](@ref), [`psnr`](@ref), [`ssim`](@ref)
"""
function snr(x::AbstractVector{T}, x₀::AbstractVector{T}) where T<:Number
    @assert length(x) == length(x₀)             # ensure same lengths
    return 20 * log(10, norm(x₀,2)/norm(x-x₀,2))
end

# SSIM between 2 arrays
"""
    ssim(x, x₀)

Wrapper for `assess_ssim` function from ImageQualityIndex.jl.

Returns the Structural Similarity Index Measure (SSIM) between the original signal/image x₀
and noisy signal/image x.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Signal with noise.
- `x₀::AbstractArray{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The SNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

ssim(x, y)
```

**See also:** [`relativenorm`](@ref), [`psnr`](@ref), [`snr`](@ref)
"""
function ssim(x::AbstractArray{T}, x₀::AbstractArray{T}) where T<:Number
    return assess_ssim(x, x₀)
end

# ----- Signal Generation -----
# Make a set of circularly shifted and noisy signals of original signal.
"""
    duplicatesignals(x, N, k[, noise=false, t=1])

Given a signal `x`, returns N shifted versions of the signal, each with shifts
of multiples of `k`. 

Setting `noise = true` allows randomly generated Gaussian noises of μ = 0, 
σ² = t to be added to the circularly shifted signals.

# Arguments
- `x::AbstractVector{T} where T<:Number`: 1D-signal to be duplicated.
- `N::Integer`:: Number of duplicated signals.
- `k::Integer`:: Circular shift size for each duplicated signal.
- `noise::Bool`: (Default: `false`) Whether or not to add Gaussian noise.
- `t::Real`: (Default: 1) Relative size of noise.

# Returns
`::Array{T}`: Duplicated signals.

# Examples
```@repl
using WaveletsExt

x = generatesignals(:blocks)
duplicatesignals(x, 5, 0)      # [x x x x x]
```

*See also:* [`generatesignals`](@ref)
"""
function duplicatesignals(x::AbstractVector{T}, 
                          N::Integer, 
                          k::Integer, 
                          noise::Bool = false, 
                          t::Real = 1) where T<:Number

    n = length(x)
    X = Array{T, 2}(undef, (n, N))
    @inbounds begin
        for i in axes(X,2)
            X[:,i] = circshift(x, k*(i-1)) 
        end
    end
    X = noise ? X + t*randn(n,N) : X
    return X
end

# Generate 6 types of signals used in popular papers.
"""
    generatesignals(fn, L)

Generates a signal of length 2ᴸ given the function symbol `fn`. Current accepted inputs 
below are based on D. Donoho and I. Johnstone in "Adapting to Unknown Smoothness via Wavelet 
Shrinkage" Preprint Stanford, January 93, p 27-28.  
- `:blocks`
- `:bumps`
- `:heavysine`
- `:doppler`
- `:quadchirp`
- `:mishmash`

The code for this function is adapted and translated based on MATLAB's Wavelet Toolbox's 
`wnoise` function.

# Arguments
- `fn::Symbol`: Type of function/signal to generate.
- `L::Integer`: (Default = 7) Size of the signal to generate. Will return a signal of size
  2ᴸ.

# Returns
`::Vector{Float64}`: Signal of length 2ᴸ.

# Examples
```repl
using WaveletsExt

generatesignals(:bumps, 8)
```
"""
function generatesignals(fn::Symbol, L::Integer = 7)
    @assert L >= 1

    t = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
    h = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2]
    
    n = 1<<L
    if fn == :blocks
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j]*(1 .+ sign.(tt .- t[j]))/2)
        end
    elseif fn == :bumps
        h = abs.(h)
        w = 0.01*[0.5, 0.5, 0.6, 1, 1, 3, 1, 1, 0.5, 0.8, 0.5]
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j] ./ (1 .+ ((tt .- t[j]) / w[j]).^4))
        end
    elseif fn == :heavysine
        x = collect(range(0, 1, length=n))
        x = 4*sin.(4*pi*x) - sign.(x .- 0.3) - sign.(0.72 .- x)
    elseif fn == :doppler
        x = collect(range(0, 1, length=n))
        ϵ = 0.05
        x = sqrt.(x.*(1 .- x)) .* sin.(2*pi*(1+ϵ) ./ (x.+ϵ))
    elseif fn == :quadchirp
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
    elseif fn == :mishmash
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
        x = x + sin.(π * (n * 0.6902) * tt)
        x = x + sin.(π * tt .* (n * 0.125 * tt))
    else
        throw(ArgumentError("Unrecognised `fn`. Type `?generatefunction` to learn more."))
    end
    return x
end

# Generate 3 classes of signals used in Saito's LDB paper.
"""
    generateclassdata(c[, shuffle])

Generates 3 classes of data given a `ClassData` struct as an input. Returns a matrix 
containing the 3 classes of signals and a vector containing their corresponding labels.

Based on N. Saito and R. Coifman in "Local Discriminant Basis and their Applications" in the
Journal of Mathematical Imaging and Vision, Vol. 5, 337-358 (1995).

# Arguments
- `c::ClassData`: Type of signal classes to generate.
- `shuffle::Bool`: (Default: `true`). Whether or not to shuffle the signals.

# Returns
- `::Matrix{Float64}`: Generated signals.
- `::Vector{Int64}`: Class corresponding to each column of generated signals.

# Examples
```@repl
using WaveletsExt

c = ClassData(:tri, 100, 100, 100)
generateclassdata(c)
```

**See also:** [`ClassData`](@ref)
"""
function generateclassdata(c::ClassData, shuffle::Bool = false)
    @assert c.s₁ >= 0
    @assert c.s₂ >= 0
    @assert c.s₃ >= 0

    if c.type == :tri
        n = 32
        i = collect(1:n)
        # Define classes
        y = Int64.(vcat(ones(Int, c.s₁), 2*ones(Int, c.s₂), 3*ones(Int, c.s₃)))
        
        # Random value generations
        u₁ = rand(Uniform(0,1), c.s₁)
        u₂ = rand(Uniform(0,1), c.s₂)
        u₃ = rand(Uniform(0,1), c.s₃)
        ϵ₁ = rand(Normal(0,1), (n, c.s₁))
        ϵ₂ = rand(Normal(0,1), (n, c.s₂))
        ϵ₃ = rand(Normal(0,1), (n, c.s₃))

        h₁ = max.(6 .- abs.(i.-7), 0)
        h₂ = max.(6 .- abs.(i.-15), 0)
        h₃ = max.(6 .- abs.(i.-11), 0)
        # Build signals
        H₁ = h₁ * u₁' + h₂ * (1 .- u₁)' + ϵ₁
        H₂ = h₁ * u₂' + h₃ * (1 .- u₂)' + ϵ₂
        H₃ = h₂ * u₃' + h₃ * (1 .- u₃)' + ϵ₃
        
        H = hcat(H₁, H₂, H₃)
    elseif c.type == :cbf
        n = 128
        ϵ = rand(Normal(0,1), (n, c.s₁+c.s₂+c.s₃))
        y = Int64.(vcat(ones(c.s₁), 2*ones(Int, c.s₂), 3*ones(Int, c.s₃)))

        d₁ = DiscreteUniform(16,32)
        d₂ = DiscreteUniform(32,96)

        # cylinder signals
        H₁ = zeros(n,c.s₁)
        a = rand(d₁,c.s₁)
        b = a+rand(d₁,c.s₁)
        η = randn(c.s₁)
        for k in 1:c.s₁
            H₁[a[k]:b[k],k]=(6+η[k])*ones(b[k]-a[k]+1)
        end

        # bell signals
        H₂ = zeros(n,c.s₂)
        a = rand(d₁,c.s₂)
        b = a+rand(d₂,c.s₂)
        η = randn(c.s₂)
        for k in 1:c.s₂
            H₂[a[k]:b[k],k]=(6+η[k])*collect(0:(b[k]-a[k]))/(b[k]-a[k])
        end

        # funnel signals
        H₃ = zeros(n,c.s₃)
        a = rand(d₁,c.s₃)
        b = a+rand(d₂,c.s₃)
        η = randn(c.s₃)
        for k in 1:c.s₃
            H₃[a[k]:b[k],k]=(6+η[k])*collect((b[k]-a[k]):-1:0)/(b[k]-a[k])
        end

        H = hcat(H₁, H₂, H₃) + ϵ
    else
        throw(ArgumentError("Invalid type. Accepted types are :tri and :cbf only."))
    end

    if shuffle
        idx = [1:(c.s₁+c.s₂+c.s₃)...]
        shuffle!(idx)
        H = H[:,idx]
        y = y[idx]
    end

    return H, y
end

end # end module