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

right(3)     # 7
```

**See also:** [`left`](@ref)
"""
right(i::Integer) = i<<1 + 1

"""
    getchildindex(idx, child)

Get the child index of a parent index `idx`.

# Arguments
- `idx::T where T<:Integer`: Index of parent node.
- `child::Symbol`: Type of child. For binary trees, available children are `:left` and
  `:right`. For quadtrees, available children are `:topleft`, `:topright`, `:bottomleft`,
  `:bottomright`.
  
# Returns
- `::T`: Index of child node.
"""
function getchildindex(idx::Integer, child::Symbol)
    @assert child ∈ [:left, :right, :topleft, :topright, :bottomleft, :bottomright]

    if child == :left
        return idx<<1
    elseif child == :right
        return idx<<1 + 1
    elseif child == :topleft
        return 4*idx - 2
    elseif child == :topright
        return 4*idx - 1
    elseif child == :bottomleft
        return 4*idx
    elseif child == :bottomright
        return 4*idx + 1
    else
        throw(ArgumentError("Invalid child $child."))
    end
end

"""
    getparentindex(idx, tree_type)

Get the parent index of the child node `idx`.

# Arguments
- `idx::T where T<:Integer`: Index of child node.
- `tree_type::Symbol`: Tree type, ie. `:binary` tree or `:quad` tree.

# Returns
- `::T`: Index of parent node.
"""
function getparentindex(idx::T, tree_type::Symbol) where T<:Integer
    @assert tree_type ∈ [:binary, :quad]

    if tree_type == :binary
        return idx>>1
    elseif tree_type == :quad
        return floor(T, (idx+2)/4)
    else
        throw(ArgumentError("Invalid tree type $tree_type."))
    end
end

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
    # Setup and Sanity check
    nₜ = length(tree)
    n = nₜ + 1
    @assert isdyadic(n)

    result = falses(n+nₜ)
    result[1] = true
    for i in eachindex(tree)
        if tree[i] == 0
            continue
        else
            @inbounds result[i] = false
            @inbounds result[getchildindex(i,:left)] = true
            @inbounds result[getchildindex(i,:right)] = true
        end
    end
    return result
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
