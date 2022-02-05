## BEST BASIS TYPES
"""
    BestBasisType

Abstract type for best basis. Current available types are:
- [`LSDB`](@ref)
- [`JBB`](@ref)
- [`BB`](@ref)
- [`SIBB`](@ref)
"""
abstract type BestBasisType end

"""
    LSDB([; cost, redundant])

Least Statistically Dependent Basis (LSDB). 

# Keyword Arguments
- `cost::LSDBCost`: (Default: `DifferentialEntropyCost()`) Cost function for LSDB.
- `redundant::Bool`: (Default: `false`) Whether the performed wavelet transform is
  redundant. Set `redundant=true` when running LSDB with redundant wavelet transforms such
  as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`JBB`](@ref), [`BB`](@ref), [`SIBB`](@ref)
"""
@with_kw struct LSDB <: BestBasisType
    cost::LSDBCost = DifferentialEntropyCost()
    redundant::Bool = false
end

"""
    JBB([; cost, redundant])

Joint Best Basis (JBB).

# Keyword Arguments
- `cost::JBBCost`: (Default: `LoglpCost(2)`) Cost function for JBB.
- `redundant::Bool`: (Default: `false`) Whether the performed wavelet transform is
  redundant. Set `redundant=true` when running LSDB with redundant wavelet transforms such
  as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`BB`](@ref), [`SIBB`](@ref)
"""
@with_kw struct JBB <: BestBasisType    # Joint Best Basis
    cost::JBBCost = LoglpCost(2)
    redundant::Bool = false
end

"""
    BB([; cost, redundant])

Standard Best Basis (BB). 

# Keyword Arguments
- `cost::BBCost`: (Default: `ShannonEntropyCost()`) Cost function for BB.
- `redundant::Bool`: (Default: `false`) Whether the performed wavelet transform is
  redundant. Set `redundant=true` when running LSDB with redundant wavelet transforms such
  as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`JBB`](@ref), [`SIBB`](@ref)
"""
@with_kw struct BB <: BestBasisType     # Individual Best Basis
    cost::BBCost = ShannonEntropyCost()
    redundant::Bool = false
end

"""
    SIBB([; cost])

Shift Invariant Best Basis (SIBB).

# Keyword Arguments
- `cost::BBCost`: (Default: `ShannonEntropyCost()`) Cost function for SIBB.

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`JBB`](@ref), 
    [`BB`](@ref)
"""
@with_kw struct SIBB <: BestBasisType   # Shift invariant best basis
    cost::BBCost = ShannonEntropyCost()
end                     


## ----- TREE COST -----
# LSDB for 1D signals
"""
    tree_costs(X, method)

Returns the cost of each node in a binary tree in order to find the best basis.

# Arguments
- `X::AbstractArray{T} where T<:AbstractFloat`: A set of decomposed signals, of sizes
  `(n,L,k)` for 1D signals or `(n,m,L,k)` for 2D signals, where:
    - `n`: Length of signal (1D) or vertical length of signal (2D).
    - `m`: Horizontal length of signal (2D).
    - `L`: Number of decomposition levels plus 1 (for standard wavelet decomposition) or
        number of nodes in the tree (for redundant transforms such as ACWT and SWT).
    - `k`: Number of signals.
- `method::BestBasisType`: Type of best basis, ie. `BB()`, `JBB()` or `LSDB()`.

!!! note
    For standard best basis (`BB()`), only one signal is processed each time, and therefore
    the inputs `X` should have dimensions `(n,L)` or `(n,m,L)` instead.

# Returns
- `Vector{T}`: A vector containing the costs at each node.

# Examples
```julia
using Wavelets, WaveletsExt

X = generatesignals(:heavisine, 6) |> x -> duplicatesignals(x, 5, 2, true)
wt = wavelet(WT.db4)
Xw = wpdall(X, wt)

tree_costs(Xw, JBB())
tree_costs(Xw, LSDB())
```

**See also:** [`bestbasistree`](@ref), [`bestbasis_treeselection`](@ref)
"""
function tree_costs(X::AbstractArray{T,3}, method::LSDB) where T<:AbstractFloat
    n, L, _ = size(X)

    if method.redundant
        costs = Vector{T}(undef, L)
        for i in eachindex(costs)
            j = getdepth(i,:binary)
            costs[i] = coefcost(X[:,i,:], method.cost) / (1<<j)
        end
    else
        costs = Vector{T}(undef, gettreelength(1<<L))
        i = 1
        for d in 0:(L-1)
            n₀ = nodelength(n, d)
            for node in 0:(1<<d-1)
                rng = (node*n₀+1):((node+1)*n₀)
                costs[i] = coefcost(X[rng, d+1, :], method.cost)
                i += 1
            end
        end
    end
    return costs
end
# LSDB for 2D signals
function tree_costs(X::AbstractArray{T,4}, method::LSDB) where T<:AbstractFloat
    n, m, L, _ = size(X)

    if method.redundant
        costs = Vector{T}(undef,L)
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            costs[i] = coefcost(X[:,:,i,:], method.cost) / (1<<(2*d))
        end
    else
        costs = Vector{T}(undef, gettreelength(1<<L,1<<L))
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            rng₁ = getrowrange(n,i)
            rng₂ = getcolrange(m,i)
            costs[i] = coefcost(X[rng₁,rng₂,d+1,:], method.cost)
        end
    end
    return costs
end

# JBB for 1D signals
function tree_costs(X::AbstractArray{T,3}, method::JBB) where T<:AbstractFloat
    # For each level, compute mean, sum of squares, and variance
    (n, L, N) = size(X)                     # L = levels if wpd, nodes if swpd
    EX = sum(X, dims = 3) ./ N              # calculate E(X)
    EX² = sum(X .^ 2, dims = 3) ./ N        # calculate E(X²)
    VarX = EX² - (EX) .^ 2                  # calculate Var(X)
    VarX = reshape(VarX, (n, L))        
    σ = VarX .^ 0.5                         # calculate σ = √Var(X)
    @assert all(σ .>= 0)

    if method.redundant
        costs = Vector{T}(undef, L)
        for i in eachindex(costs)
            j = floor(Integer, log2(i))
            costs[i] = coefcost(σ[:, i], method.cost) / (1<<j)
        end
    else
        costs = Vector{T}(undef, gettreelength(1<<L))
        i = 1   # iterates over the nodes for the costs variable
        for lvl in 0:(L-1)
            n₀ = nodelength(n, lvl)
            for node in 0:(2^lvl-1)
                rng = (node * n₀ + 1):((node + 1) * n₀)
                coef = σ[rng, lvl+1]
                costs[i] = coefcost(coef, method.cost)
                i += 1
            end
        end
    end
    return costs
end
# JBB for 2D signals
function tree_costs(X::AbstractArray{T,4}, method::JBB) where T<:AbstractFloat
    # For each level, compute mean, sum of squares, and variance
    n,m,L,N = size(X)
    EX = sum(X, dims=4) / N
    EX² = sum(X.^2, dims=4) / N
    VarX = EX² - EX.^2 |> y -> dropdims(y, dims=4)
    σ = sqrt.(VarX)
    @assert all(σ .≥ 0)

    if method.redundant
        costs = Vector{T}(undef,L)
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            costs[i] = coefcost(σ[:,:,i], method.cost) / (1<<(2*d))
        end
    else
        costs = Vector{T}(undef,gettreelength(1<<L,1<<L))
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            rng₁ = getrowrange(n,i)
            rng₂ = getcolrange(m,i)
            costs[i] = coefcost(σ[rng₁,rng₂,d+1], method.cost)
        end
    end
    return costs
end

# Standard Best Basis for 1D signals
function tree_costs(X::AbstractArray{T,2}, method::BB) where T<:AbstractFloat
    nrm = norm(X[:,1])
    L = size(X, 2)                      # count of levels if wpd, nodes if swpd
    n = size(X,1)

    if method.redundant                # swpd
        costs = Vector{T}(undef, L)
        for i in eachindex(costs)
            j = floor(Integer, log2(i))
            costs[i] = coefcost(X[:, i], method.cost, nrm) / (1<<j)
        end
    else
        costs = Vector{T}(undef, 2^L - 1)
        i = 1
        for lvl in 0:(L-1)
            n₀ = nodelength(n, lvl)
            for node in 0:(2^lvl-1)
                rng = (node * n₀ + 1):((node + 1) * n₀)
                costs[i] = coefcost(X[rng, lvl+1, :], method.cost, nrm)
                i += 1
            end
        end
    end
    return costs
end
# Standard Best Basis for 2D signals
function tree_costs(X::AbstractArray{T,3}, method::BB) where T<:AbstractFloat
    nrm = norm(X[:,:,1])
    n,m,L = size(X)

    if method.redundant
        costs = Vector{T}(undef,L)
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            costs[i] = coefcost(X[:,:,i], method.cost, nrm) / (1<<(2*d))
        end
    else
        costs = Vector{T}(undef, gettreelength(1<<L,1<<L))
        for i in eachindex(costs)
            d = getdepth(i,:quad)
            rng₁ = getrowrange(n,i)
            rng₂ = getcolrange(m,i)
            costs[i] = coefcost(X[rng₁,rng₂,d+1], method.cost)
        end
    end
    return costs
end

# SIWPD for 1D signals
"""
    tree_costs(y, tree, method)

Computes the cost for each node from the SIWPD decomposition.

# Arguments
- `y::AbstractArray{T,2} where T<:Number`: A SIWPD decomposed signal.
- `tree::AbstractVector{BitVector}`: The full SIWPD tree.
- `method::SIBB`: The `SIBB()` method.

# Returns
- `Vector{Vector{Union{T,Nothing}}}`: SIWPD best basis tree.

!!! warning
    Current implementation works but is unstable, ie. we are still working on better
    syntax/more optimized computations/better data structure.
"""
function tree_costs(y::AbstractArray{T,2}, tree::AbstractVector{BitVector}, 
        method::SIBB) where T<:Number

    nn = length(tree)                           
    ns = size(y,1)                              
    @assert size(y,2) == nn                     
    tree_costs = Vector{Vector{Union{T,Nothing}}}(undef, nn)
    nrm = norm(y[:,1])                          

    for i in eachindex(tree)
        level = floor(Int, log2(i))
        len = nodelength(ns, level)
        # number of nodes corresponding to Ω(i,j)
        costs = Vector{Union{AbstractFloat,Nothing}}(nothing, length(tree[i]))
        for j in eachindex(tree[i])
            if tree[i][j]
                shift = j-1                     # current shift
                nstart = shift*len + 1
                nend = (shift+1) * len
                costs[j] = coefcost(y[nstart:nend,i], method.cost, nrm)
            end
        end
        tree_costs[i] = costs
    end
    return tree_costs
end
