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
    SIBB,
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
    ..DWT,
    ..SIWPD

include("bestbasis_costs.jl")
include("bestbasis_tree.jl")

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

# SIWPD tree selection
"""
    bestbasis_treeselection(costs, tree)

Best basis tree selection on SIWPD.

# Arguments
- `costs::AbstractVector{Tc} where Tc<:AbstractVector{<:Union{Number, Nothing}}`: Cost of
  each node.
- `tree::AbstractVector{Tt} where Tt<:BitVector`: SIWPD tree.

!!! warning
    Current implementation works but is unstable, ie. we are still working on better
    syntax/more optimized computations/better data structure.
"""
function bestbasis_treeselection(costs::AbstractVector{Tc}, 
        tree::AbstractVector{Tt}) where 
        {Tc<:AbstractVector{<:Union{Number,Nothing}}, Tt<:BitVector}

    @assert length(costs) == length(tree)
    bt = deepcopy(tree)
    bc = deepcopy(costs)
    nn = length(tree)
    for i in reverse(eachindex(bt))
        if getchildindex(i,:left) > nn                        # current node is at bottom level
            continue
        end
        level = floor(Int, log2(i))
        for j in eachindex(bt[i])       # iterate through all available shifts
            if !bt[i][j]                # node of current shift does not exist
                continue
            end
            @assert bt[getchildindex(i,:left)][j] == 
                    bt[getchildindex(i,:right)][j] == 
                    bt[getchildindex(i,:left)][j+1<<level] == 
                    bt[getchildindex(i,:right)][j+1<<level] == true || continue
            # child cost of current shift of current node
            cc = bc[getchildindex(i,:left)][j] + bc[getchildindex(i,:right)][j]      
            # child cost of circshift of current node               
            scc = bc[getchildindex(i,:left)][j+1<<level] + 
                  bc[getchildindex(i,:right)][j+1<<level]  
            mincost = min(cc, scc, bc[i][j])
            # mincost=parent, delete subtrees of both cc & scc                    
            if mincost == bc[i][j]          
                delete_subtree!(bt, getchildindex(i,:left), j)
                delete_subtree!(bt, getchildindex(i,:right), j)
                delete_subtree!(bt, getchildindex(i,:left), j+1<<level)
                delete_subtree!(bt, getchildindex(i,:right), j+1<<level)
            # mincost=cc, delete subtrees of scc
            elseif mincost == cc                                             
                bc[i][j] = mincost
                delete_subtree!(bt, getchildindex(i,:left), j+1<<level)
                delete_subtree!(bt, getchildindex(i,:right), j+1<<level)
            # mincost=scc, delete subtrees of cc
            else                                                                    
                bc[i][j] = mincost
                delete_subtree!(bt, getchildindex(i,:left), j)
                delete_subtree!(bt, getchildindex(i,:right), j)
            end
        end
    end
    # ensure only 1 node selected from each Ω(i,j)
    @assert all(map(node -> sum(node), bt) .<= 1)  
    return bt
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

function delete_subtree!(bt::AbstractVector{BitVector}, i::Integer, j::Integer)
    @assert 1 <= i <= length(bt)
    level = floor(Int, log2(i))
    bt[i][j] = false
    if (getchildindex(i,:left)) < length(bt)          # current node can contain subtrees
        if bt[getchildindex(i,:left)][j]              # left child of current shift
            delete_subtree!(bt, getchildindex(i,:left), j)
        end
        if bt[getchildindex(i,:right)][j]            # right child of current shift
            delete_subtree!(bt, getchildindex(i,:right), j)
        end
        if bt[getchildindex(i,:left)][j+1<<level]     # left child of added shift
            delete_subtree!(bt, getchildindex(i,:left), j+1<<level)
        end
        if bt[getchildindex(i,:right)][j+1<<level]   # right child of added shift
            delete_subtree!(bt, getchildindex(i,:right), j+1<<level)
        end
    end
    return nothing
end


## BEST BASIS TREES
# Best basis tree for LSDB
"""
    bestbasistree(X[, method])

Extension to the best basis tree function from Wavelets.jl. Given a set of decomposed
signals, returns different types of best basis trees based on the methods specified.
Available methods are the joint best basis ([`JBB`](@ref)), least statistically dependent
basis ([`LSDB`](@ref)), individual regular best basis ([`BB`](@ref)), and shift-invariant
best basis ([`SIBB`](@ref)).

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
# SIWPD Best basis tree
# TODO: find a way to compute bestbasis_tree without input d
"""
    bestbasistree(y, d, method)

Computes the best basis tree for the shift invariant wavelet packet decomposition (SIWPD).

# Arguments
- `y::AbstractArray{T,2} where T<:Number`: A SIWPD decomposed signal.
- `d::Integer`: The number of depth computed for the decomposition.
- `method::SIBB`: The `SIBB()` method.

# Returns
- `Vector{BitVector}`: SIWPD best basis tree.

!!! warning
    Current implementation works but is unstable, ie. we are still working on better
    syntax/more optimized computations/better data structure.
"""
function Wavelets.Threshold.bestbasistree(y::AbstractArray{T,2}, 
                                          d::Integer, 
                                          method::SIBB) where T<:Number                                           
    
    nn = size(y,2) 
    L = maxtransformlevels((nn+1)÷2)
    ns = size(y,1)
    tree = makesiwpdtree(ns, L, d)
    costs = tree_costs(y, tree, method)
    besttree = bestbasis_treeselection(costs, tree)
    return besttree
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