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
    coefcost,
    # tree cost
    tree_costs,
    # best tree selection
    bestbasis_treeselection,
    # best basis types
    BestBasisType,
    LSDB,
    JBB,
    BB,
    SIBB,
    # best basis tree
    bestbasis_tree

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


## COST COMPUTATION
"""
Cost function abstract type.

**See also:** [`LSDBCost`](@ref), [`JBBCost`](@ref), [`BBCost`](@ref)
"""
abstract type CostFunction end

"""
    LSDBCost <: CostFunction

Cost function abstract type specifically for LSDB.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`BBCost`](@ref)
"""
abstract type LSDBCost <: CostFunction end

"""
    JBBCost <: CostFunction
    
Cost function abstract type specifically for JBB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref), [`BBCost`](@ref)
"""
abstract type JBBCost <: CostFunction end

"""
    BBCost <: CostFunction
    
Cost function abstract type specifically for BB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref), [`JBBCost`](@ref)
"""
abstract type BBCost <: CostFunction end

@doc raw"""
    LoglpCost <: JBBCost
    
``\log \ell^p`` information cost used for JBB. Typically, we set `p=2` as in 
Wickerhauser's original algorithm.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`NormCost`](@ref)
"""
@with_kw struct LoglpCost <: JBBCost 
    p::Number = 2
end

@doc raw"""
    NormCost <: JBBCost
    
``p``-norm information cost used for JBB.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`LoglpCost`](@ref)
"""
@with_kw struct NormCost <: JBBCost 
    p::Number = 1
end

"""
    DifferentialEntropyCost <: LSDBCost

Differential entropy cost used for LSDB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref)
"""
struct DifferentialEntropyCost <: LSDBCost end

"""
    ShannonEntropyCost <: LSDBCost

Shannon entropy cost used for BB.

**See also:** [`CostFunction`](@ref), [`BBCost`](@ref), 
    [`LogEnergyEntropyCost`](@ref)
"""
struct ShannonEntropyCost <: BBCost end

"""
    LogEnergyEntropyCost <: LSDBCost

Log energy entropy cost used for BB.

**See also:** [`CostFunction`](@ref), [`BBCost`](@ref), 
    [`ShannonEntropyCost`](@ref)
"""
struct LogEnergyEntropyCost <: BBCost end

# cost functions for individual best basis algorithm
function coefcost(x::T, et::ShannonEntropyCost, nrm::T) where T<:AbstractFloat
    s = (x/nrm)^2
    if s == 0.0
        return -zero(T)
    else
        return -s*log(s)
    end
end

function coefcost(x::T, et::LogEnergyEntropyCost, nrm::T) where T<:AbstractFloat
    s = (x/nrm)^2
    if s == 0.0
        return -zero(T)
    else
        return -log(s)
    end
end

function coefcost(x::AbstractArray{T}, et::BBCost, nrm::T=norm(x)) where 
        T<:AbstractFloat

    @assert nrm >= 0
    sum = zero(T)
    nrm == sum && return sum
    for i in eachindex(x)
        @inbounds sum += coefcost(x[i], et, nrm)
    end
    return sum
end

# cost functions for joint best basis (JBB)
function coefcost(x::AbstractArray{T}, et::LoglpCost) where T<:Number
    xᵖ = abs.(x) .^ et.p
    return sum(log.(xᵖ))
end

function coefcost(x::AbstractArray{T}, et::NormCost) where T<:Number
    return norm(x, et.p)
end

# cost functions for least statistically dependent basis (LSDB)
function coefcost(x::AbstractVector{T}, et::DifferentialEntropyCost) where 
        T<:AbstractFloat

    N = length(x)                           # length of vector x
    M = 50                                  # arbitrary large number M

    nbins = ceil(Int64, (30 * N)^(1/5))     # number of bins per histogram
    mbins = ceil(Int64, M/nbins)            # number of histograms calculated

    σ = std(x)                              # standard deviation of x
    # setup range for ASH
    s = 0.5                     
    δ = (maximum(x) - minimum(x) + σ)/((nbins+1)*mbins-1)       
    rng = (minimum(x) - s*σ):δ:(maximum(x) + s*σ)     
    # compute ASH          
    epdf = ash(x, rng=rng, m=mbins, kernel=Kernels.triangular)
    ent = 0
    for k in 1:N
        ent -= (1/N) * log(AverageShiftedHistograms.pdf(epdf, x[k]))
    end
    return ent
end

function coefcost(x::AbstractArray{T,2}, et::DifferentialEntropyCost) where 
        T<:Number

    cost = 0
    for i in axes(x,1)
        @inbounds cost += coefcost(x[i,:], DifferentialEntropyCost())
    end
    return cost
end


## BEST BASIS TYPES
"""
Abstract type for best basis. Current available types are:
- [`LSDB`](@ref)
- [`JBB`](@ref)
- [`BB`](@ref)
- [`SIBB`](@ref)
"""
abstract type BestBasisType end

"""
    LSDB([; cost=DifferentialEntropyCost(), redundant=false])

Least Statistically Dependent Basis (LSDB). Set `redundant=true` when running
LSDB with redundant wavelet transforms such as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`JBB`](@ref), [`BB`](@ref), 
    [`SIBB`](@ref)
"""
@with_kw struct LSDB <: BestBasisType
    cost::LSDBCost = DifferentialEntropyCost()
    redundant::Bool = false
end

"""
    JBB([; cost=LoglpCost(2), redundant=false])

Joint Best Basis (JBB). Set `redundant=true` when running JBB with redundant 
wavelet transforms such as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`BB`](@ref), 
    [`SIBB`](@ref)
"""
@with_kw struct JBB <: BestBasisType    # Joint Best Basis
    cost::JBBCost = LoglpCost(2)
    redundant::Bool = false
end

"""
    BB([; cost=LoglpCost(2), redundant=false])

Best Basis (BB). Set `redundant=true` when running BB with redundant wavelet 
transforms such as SWT or ACWT.

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`JBB`](@ref), 
    [`SIBB`](@ref)
"""
@with_kw struct BB <: BestBasisType     # Individual Best Basis
    cost::BBCost = ShannonEntropyCost()
    redundant::Bool = false
end

"""
    SIBB([; cost=ShannonEntropyCost()])

Shift Invariant Best Basis (SIBB).

**See also:** [`BestBasisType`](@ref), [`LSDB`](@ref), [`JBB`](@ref), 
    [`BB`](@ref)
"""
@with_kw struct SIBB <: BestBasisType   # Shift invariant best basis
    cost::BBCost = ShannonEntropyCost()
end                     


## TREE COST
"""
    tree_costs(X, method)

Returns the cost of each node in a binary tree in order to find the best basis.

**See also:** [`bestbasistree`](@ref), [`bestbasis_treeselection`](@ref)
"""
function tree_costs(X::AbstractArray{T,3}, method::LSDB) where T<:AbstractFloat
    L = size(X, 2)
    n = size(X, 1)

    if method.redundant
        costs = Vector{T}(undef, L)
        for i in eachindex(costs)
            j = floor(Integer, log2(i))
            costs[i] = coefcost(X[:,i,:], method.cost) / (1<<j)
        end
    else
        costs = Vector{T}(undef, 2^L - 1)
        i = 1
        for lvl in 0:(L-1)
            n₀ = nodelength(n, lvl)
            for node in 0:(2^lvl-1)
                rng = (node * n₀ + 1):((node + 1) * n₀)
                costs[i] = coefcost(X[rng, lvl+1, :], method.cost)
                i += 1
            end
        end
    end
    return costs
end

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
        costs = Vector{T}(undef, 2^L - 1)
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


## BEST TREE SELECTION
"""
    bestbasis_treeselection(costs, n[, type=:min])

Computes the best tree based on the given cost vector.

**See also:** [`bestbasistree`](@ref), [`tree_costs`](@ref)
"""
function bestbasis_treeselection(costs::AbstractVector{T}, n::Integer,
        type::Symbol=:min) where T<:AbstractFloat

    @assert length(costs) <= 2*n - 1
    L = floor(Int, log2(length(costs)))
    bt = maketree(n, L, :full)
    if type == :min
        @inbounds begin
            for i in reverse(1:(1<<L-1))
                childcost = costs[getchildindex(i,:left)] + costs[getchildindex(i,:right)]
                if childcost < costs[i]     # child cost < parent cost
                    costs[i] = childcost
                else
                    delete_subtree!(bt, i)
                end
            end
        end
    elseif type == :max
        @inbounds begin
            for i in reverse(1:(1<<L-1))
                childcost = costs[getchildindex(i,:left)] + costs[getchildindex(i,:right)]
                if childcost > costs[i]     # child cost < parent cost
                    costs[i] = childcost
                else
                    delete_subtree!(bt, i)
                end
            end
        end
    else
        throw(ArgumentError("Accepted types are :min and :max only"))
    end
    return bt
end

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

# deletes subtree due to inferior cost
function delete_subtree!(bt::BitVector, i::Integer)
    @assert 1 <= i <= length(bt)
    bt[i] = false
    if (getchildindex(i,:left)) < length(bt)
        if bt[getchildindex(i,:left)]
            delete_subtree!(bt, getchildindex(i,:left))
        end
        if bt[getchildindex(i,:right)]
            delete_subtree!(bt, getchildindex(i,:right))
        end
    end
    return nothing
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
"""
    bestbasistree(X[, method])

Extension to the best basis tree function from Wavelets.jl. Given a set of 
decomposed signals, returns different types of best basis trees based on the 
methods specified. Available methods are the joint best basis ([`JBB`](@ref)), 
least statistically dependent basis ([`LSDB`](@ref)), individual regular 
best basis ([`BB`](@ref)), and shift-invariant best basis ([`SIBB`](@ref)).

# Examples
```julia
bestbasistree(X, JBB())

bestbasistree(X, SIBB())
```

**See also:** [`bestbasiscoef`](@ref)
"""
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3},                
        method::LSDB) where T<:AbstractFloat
    
    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, size(X,1))
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3}, 
        method::JBB) where T<:AbstractFloat

    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, size(X,1))
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3},                
        method::BB) where T<:AbstractFloat
    
    n = size(X,1)
    besttree = falses(n-1, size(X,3))
    @inbounds begin
        for i in axes(besttree,2)
            costs = tree_costs(X[:,:,i], method)
            besttree[:,i] = bestbasis_treeselection(costs, n)
        end
    end
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,2}, 
        method::BB) where T<:AbstractFloat

    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, size(X,1))
    return besttree
end

function Wavelets.Threshold.bestbasistree(y::AbstractArray{T,2}, d::Integer, 
        method::SIBB) where T<:Number                                           # TODO: find a way to compute bestbasis_tree without input d
        
    nn = size(y,2) 
    L = maxtransformlevels((nn+1)÷2)
    ns = size(y,1)
    tree = makesiwpdtree(ns, L, d)
    costs = tree_costs(y, tree, method)
    besttree = bestbasis_treeselection(costs, tree)
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3}; 
        method::BestBasisType=JBB()) where T<:AbstractFloat
    return bestbasistree(X, method)
end

end # end of module