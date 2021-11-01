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
    maketree(x[, s])
    maketree(n, m, L[, s])

Build quadtree for 2D wavelet transform. Indexing of the tree are as follows:

```
Level 0                 Level 1                 Level 2             ...
_________________       _________________       _________________
|               |       |   2   |   3   |       |_6_|_7_|10_|11_|
|       1       |       |_______|_______|       |_8_|_9_|12_|13_|   ...
|               |       |   4   |   5   |       |14_|15_|18_|19_|
|_______________|       |_______|_______|       |16_|17_|20_|21_|
```

# Arguments
- `x::AbstractMatrix{T} where T<:Number`: Input array.
- `n::Integer`: Number of rows in `x`.
- `m::Integer`: Number of columns in `x`.
- `L::Integer`: Number of decomposition levels.
- `s::Symbol`: (Default: `:full`) Type of quadtree. Available types are `:full` and `:dwt`.

# Returns
`::BitVector`: Quadtree representation.

# Examples
```@repl
using Wavelets, WaveletsExt

x = randn(16,16)
maketree(x)
```
"""
function Wavelets.Util.maketree(x::AbstractMatrix{T}, s::Symbol = :full) where T<:Number
    return maketree(size(x,1), size(x,2), maxtransformlevels(x), s)
end

function Wavelets.Util.maketree(n::Integer, m::Integer, L::Integer, s::Symbol = :full)
    L₀ = maxtransformlevels(min(n,m))
    @assert 0 ≤ L ≤ L₀
    # TODO: Find a mathematical formula to define this rather than sum things up to speed up
    # TODO: process.
    nq = sum(4 .^(0:(L₀-1)))

    # Construct quadtree
    tree = falses(nq)

    # Fill in true values depending on input s
    if s == :full
        rng = 4 .^ (0:(L-1)) |> sum |> x -> 1:x
        for i in rng
            @inbounds tree[i] = true
        end
    elseif s == :dwt
        tree[1] = true     # Root node
        for i in 0:(L-2)
            # TODO: Find a mathematical formula to define this
            idx = 4 .^ (0:i) |> sum |> x -> x+1     # Subsequent LL subspace nodes
            @inbounds tree[idx] = true
        end
    else
        throw(ArgumentError("Unknown symbol."))
    end
    return tree
end

# Get level of a particular index in a tree
"""
    getdepth(idx, tree_type)

Get depth of `idx` in a binary tree or quadtree.

# Arguments
- `idx::T where T<:Integer`: Index of a node.
- `tree_type::Symbol`: Tree type. Supported types are `:binary` and `:quad` trees.

# Returns
`::T`: Depth of node `idx`.

# Examples
```@repl
using WaveletsExt

getdepth(1,:binary)     # 0
getdepth(3,:binary)     # 1
getdepth(8,:binary)     # 3

getdepth(1,:quad)       # 0
getdepth(3,:quad)       # 1
getdepth(8,:quad)       # 2
```

**See also:** [`makequadtree`](@ref)
"""
function getdepth(idx::T, tree_type::Symbol) where T<:Integer
    @assert idx > 0
    @assert tree_type ∈ [:binary, :quad]

    if tree_type == :binary
        return floor(T, log2(idx))
    elseif tree_type == :quad
        return floor(T, log(4, 3*idx-2))
    else
        throw(ArgumentError("Unsupported tree type $tree_type."))
    end
end
