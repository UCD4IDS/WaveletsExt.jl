module BestBasis
export 
    # cost functions
    CostFunction,
    LSDBCost,
    JBBCost,
    BBCost,
    LoglpCost,
    NormCost,
    DifferentialEntropyCost,
    ShannonEntropyCost,
    LogEnergyEntropyCost,
    # tree cost
    tree_costs,
    # best basis types
    BestBasisType,
    LSDB,
    JBB,
    BB,
    # best basis tree
    bestbasistreeall

using 
    Wavelets, 
    LinearAlgebra,
    Statistics, 
    AverageShiftedHistograms,
    Parameters

using 
    ..Utils,
    ..DWT

include("bestbasis/bestbasis_costs.jl")
include("bestbasis/bestbasis_tree.jl")

## ----- BEST TREE SELECTION -----
# Tree selection for 1D signals
"""
    bestbasis_treeselection(costs, n[, type])
    bestbasis_treeselection(costs, n, m[, type])

Computes the best basis tree based on the given cost vector.

# Arguments
- `costs::AbstractVector{T}`: Vector containing costs for each node.
- `n::Integer`: Length of signals (for 1D cases) or vertical length of signals (for 2D
  cases).
- `m::Integer`: Horizontal length of signals (for 2D cases).
- `type::Symbol`: (Default: `:min`) Criterion used to select the best tree. Supported types
  are `:min` and `:max`. Eg. Setting `type = :min` results in a basis tree with the lowest
  cost to be selected.

# Returns
- `::BitVector`: Best basis tree selected based on cost.

**See also:** [`bestbasistree`](@ref), [`tree_costs`](@ref)
"""
function bestbasis_treeselection(costs::AbstractVector{T}, 
                                 n::Integer, 
                                 type::Symbol = :min) where T<:AbstractFloat
    k = length(costs)
    @assert k ≤ gettreelength(2*n)
    @assert type ∈ [:min, :max] || throw(ArgumentError("Unsupported type $type."))
    L = getdepth(k,:binary)
    tree = maketree(n, L, :full)
    for i in reverse(eachindex(tree))
        if tree[i]                  # Check costs if node i exists
            # Get parent and children costs
            @inbounds pc = costs[i]
            @inbounds cc = costs[getchildindex(i,:left)] + costs[getchildindex(i,:right)]
            if type == :min && cc < pc          # Child cost < parent cost
                @inbounds costs[i] = cc
            elseif type == :max && cc > pc      # Child cost > parent cost
                @inbounds costs[i] = cc
            else
                delete_subtree!(tree, i, :binary)
            end
        end
    end
    @assert isvalidtree(zeros(n), tree)
    return tree
end
# Tree selection for 2D signals
function bestbasis_treeselection(costs::AbstractVector{T},
                                 n::Integer, m::Integer,
                                 type::Symbol = :min) where T<:AbstractFloat
    k = length(costs)
    @assert k ≤ gettreelength(2*n,2*m)
    @assert type ∈ [:min, :max] || throw(ArgumentError("Unsupported type $type."))
    L = getdepth(k,:quad)
    tree = maketree(n, m, L, :full)
    for i in reverse(eachindex(tree))
        if tree[i]
            # Get parent and children costs
            @inbounds pc = costs[i]
            @inbounds cc = costs[getchildindex(i,:topleft)] + costs[getchildindex(i,:topright)] +
                        costs[getchildindex(i,:bottomleft)] + costs[getchildindex(i,:bottomright)]
            if type == :min && cc < pc
                @inbounds costs[i] = cc
            elseif type == :max && cc > pc
                @inbounds costs[i] = cc
            else
                delete_subtree!(tree, i, :quad)
            end
        end
    end
    @assert isvalidtree(zeros(n,m), tree)
    return tree
end

# Deletes subtree due to inferior cost
"""
    delete_subtree!(bt, i, tree_type)

Deletes a subtree of the entire tree due to children's inferior costs.

# Arguments
- `bt::BitVector`: Tree.
- `i::Integer`: Root of the subtree to be deleted.
- `tree_type::Symbol`: Type of tree (`:binary` or `:quad`).

# Returns
`bt::BitVector`: Tree with deleted subtree.

**See also:** [`bestbasistree`](@ref), [`bestbasis_treeselection`](@ref)
"""
function delete_subtree!(bt::BitVector, i::Integer, tree_type::Symbol)
    @assert 1 ≤ i ≤ length(bt)
    @assert tree_type ∈ [:binary, :quad]
    children = tree_type == :binary ? [:left, :right] : [:topleft, :topright, :bottomleft, :bottomright]
    bt[i] = false
    for c in children
        # Delete child subtree if exist
        if (getchildindex(i,c) ≤ length(bt)) && bt[getchildindex(i,c)]
            delete_subtree!(bt, getchildindex(i,c), tree_type)
        end
    end
    return bt
end

## BEST BASIS TREES
# Best basis tree for LSDB
"""
    bestbasistree(X[, method])

Extension to the best basis tree function from Wavelets.jl. Given a set of decomposed
signals, returns different types of best basis trees based on the methods specified.
Available methods are the joint best basis ([`JBB`](@ref)), least statistically dependent
basis ([`LSDB`](@ref)), and individual regular best basis ([`BB`](@ref)).

# Arguments
- `X::AbstractArray{T} where T<:AbstractFloat`: A set of decomposed signals, of sizes
  `(n,L,k)` for 1D signals or `(n,m,L,k)` for 2D signals, where:
    - `n`: Length of signal (1D) or vertical length of signal (2D).
    - `m`: Horizontal length of signal (2D).
    - `L`: Number of decomposition levels plus 1 (for standard wavelet decomposition) or
      number of nodes in the tree (for redundant transforms such as ACWT and SWT).
    - `k`: Number of signals.
- `method::BestBasisType`: Type of best basis, ie. `BB()`, `JBB()` or `LSDB()`.

!!! tip
    For standard best basis (`BB()`), this current function can only process one signal at a
    time, ie. the input `X` should have dimensions `(n,L)` or `(n,m,L)`. To process multiple
    signals using one function, see [`bestbasistreeall`](@ref).

# Returns
- `::BitVector`: Best basis tree.

# Examples
```julia
using Wavelets, WaveletsExt

X = generatesignals(:heavisine, 6) |> x -> duplicatesignals(x, 5, 2, true)
wt = wavelet(WT.db4)
Xw = wpdall(X, wt)

bestbasistree(Xw, JBB())
bestbasistree(Xw, LSDB())
```

**See also:** [`getbasiscoef`](@ref), [`getbasiscoefall`](@ref), [`tree_costs`](@ref),
[`delete_subtree!`](@ref)
"""
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T}, method::LSDB) where 
                                          T<:AbstractFloat
    @assert 3 ≤ ndims(X) ≤ 4                # Compatible for 1D and 2D decomposed signals
    sz = size(X)[1:end-2]                   # Signal size
    costs = tree_costs(X, method)
    tree = bestbasis_treeselection(costs, sz...)
    return tree
end
# Best basis tree for JBB
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T}, method::JBB) where 
                                          T<:AbstractFloat
    @assert 3 ≤ ndims(X) ≤ 4                # Compatible for 1D and 2D decomposed signals
    sz = size(X)[1:end-2]                   # Signal size
    costs = tree_costs(X, method)
    tree = bestbasis_treeselection(costs, sz...)
    return tree
end
# Standard best basis tree
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T}, method::BB) where 
    T<:AbstractFloat
    @assert 2 ≤ ndims(X) ≤ 3                # Compatible for 1D and 2D decomposed signal
    sz = size(X)[1:end-1]                   # Signal size
    costs = tree_costs(X, method)
    tree = bestbasis_treeselection(costs, sz...)
    return tree
end

# Default best basis tree search
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T}, 
                                          method::BestBasisType = JBB()) where 
                                          T<:AbstractFloat
    return bestbasistree(X, method)
end

"""
    bestbasistreeall(X, method)

Compute the standard best basis tree of a set of signals.

# Arguments
- `X::AbstractArray{T} where T<:AbstractFloat`: A set of decomposed signals, of sizes
  `(n,L,k)` for 1D signals or `(n,m,L,k)` for 2D signals, where:
    - `n`: Length of signal (1D) or vertical length of signal (2D).
    - `m`: Horizontal length of signal (2D).
    - `L`: Number of decomposition levels plus 1 (for standard wavelet decomposition) or
        number of nodes in the tree (for redundant transforms such as ACWT and SWT).
    - `k`: Number of signals.
- `method::BB`: Standard best basis method, eg. `BB()`.

# Returns
- `::BitMatrix`: `(nₜ,k)` matrix where each column corresponds to a tree.

# Examples
```julia
using Wavelets, WaveletsExt

X = generatesignals(:heavisine, 6) |> x -> duplicatesignals(x, 5, 2, true)
wt = wavelet(WT.db4)

Xw = wpdall(X, wt)
bestbasistreeall(Xw, BB())

Xw = swpdall(X, wt)
bestbasistree(Xw, BB(redundant=true))
```

**See also:** [`bestbasistree`](@ref)
"""
function bestbasistreeall(X::AbstractArray{T}, method::BB) where T<:AbstractFloat
    @assert 3 ≤ ndims(X) ≤ 4                # Compatible for 1D and 2D decomposed signals
    sz = size(X)[1:end-2]                   # Signal size
    k = size(X)[end]                        # Number of signals
    trees = falses(gettreelength(sz...), k) # Allocate trees
    @views for (i, Xᵢ) in enumerate(eachslice(X, dims=ndims(X)))
        @inbounds trees[:,i] = bestbasistree(Xᵢ, method)
    end
    return trees
end

end # end of module