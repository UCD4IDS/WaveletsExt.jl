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

Utils.packet(0, 0, 8)       # 1:8
Utils.packet(2, 1, 8)       # 3:4
```

Translated from Wavelab850 by Zeng Fung Liew.
"""
function packet(d::Integer, b::Integer, n::Integer)
    npack = 1 << d                                  # Number of blocks in current level
    p = (b * (n÷npack) + 1) : ((b+1) * (n÷npack))   # Range of specified block
    return p
end

# Shift decomposed into BitVector
"""
    main2depthshift(sm, L)

Given the overall shift `sm`, compute the cumulative shift at each depth. Useful for
computing the shift based inverse redundant wavelet transforms.

# Arguments
- `sm::Integer`: Overall shift.
- `L::Integer`: The total number of depth.

# Returns
`::Vector{Int}`: Vector where each entry `i` describes the shift at depth `i-1`.

# Examples
```
using WaveletsExt

Utils.main2depthshift(10, 4)      # [0, 0, 2, 2, 10]
Utils.main2depthshift(5, 5)       # [0, 1, 1, 5, 5, 5]
```
"""
function main2depthshift(sm::Integer, L::Integer)
    sb = Array{Bool,1}(undef, L)
    digits!(sb, sm, base=2)         # Compute if a shift is necessary at each depth
    d = collect(0:(L-1))            # Collect all depths
    sd = sb .<< d |> cumsum         # Compute overall shift at each depth
    pushfirst!(sd, 0)               # At depth 0, there will be no shift
    return sd
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

include("utils_metrics.jl")
include("utils_dataset.jl")

end # end module