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
    bestbasis_tree,
    # best basis expansion coefficients
    bestbasiscoef

using 
    Wavelets, 
    LinearAlgebra,
    Statistics, 
    AverageShiftedHistograms,
    Parameters

using 
    ..Utils,
    ..WPD,
    ..SIWPD


## COST COMPUTATION
abstract type CostFunction end
abstract type LSDBCost <: CostFunction end
abstract type JBBCost <: CostFunction end
abstract type BBCost <: CostFunction end
@with_kw struct LoglpCost <: JBBCost 
    p::Number = 2
end
@with_kw struct NormCost <: JBBCost 
    p::Number = 1
end
struct DifferentialEntropyCost <: LSDBCost end
struct ShannonEntropyCost <: BBCost end  
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
abstract type BestBasisType end
@with_kw struct LSDB <: BestBasisType   # Least Statistically Dependent Basis
    cost::LSDBCost = DifferentialEntropyCost()
    redundant::Bool = false
end
@with_kw struct JBB <: BestBasisType    # Joint Best Basis
    cost::JBBCost = LoglpCost(2)
    redundant::Bool = false
end                                     
@with_kw struct BB <: BestBasisType     # Individual Best Basis
    cost::BBCost = ShannonEntropyCost()
    redundant::Bool = false
end                                     
@with_kw struct SIBB <: BestBasisType   # Shift invariant best basis
    cost::BBCost = ShannonEntropyCost()
end                     


## TREE COST
"""
    tree_costs(X, method)

Returns the cost of each node in a binary tree in order to find the best basis.
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
    bestbasis_treeselection(costs, L[, type=:min])

Computes the best tree based on the given cost vector.
"""
function bestbasis_treeselection(costs::AbstractVector{T}, L::Integer,
        type::Symbol=:min) where T<:AbstractFloat

    @assert length(costs) == 1<<(L+1) - 1
    bt = trues(1<<L - 1)
    if type == :min
        @inbounds begin
            for i in reverse(eachindex(bt))
                childcost = costs[left(i)] + costs[right(i)]
                if childcost < costs[i]     # child cost < parent cost
                    costs[i] = childcost
                else
                    delete_subtree!(bt, i)
                end
            end
        end
    elseif type == :max
        @inbounds begin
            for i in reverse(eachindex(bt))
                childcost = costs[left(i)] + costs[right(i)]
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
        if left(i) > nn                        # current node is at bottom level
            continue
        end
        level = floor(Int, log2(i))
        for j in eachindex(bt[i])       # iterate through all available shifts
            if !bt[i][j]                # node of current shift does not exist
                continue
            end
            @assert bt[left(i)][j] == bt[right(i)][j] == 
                bt[left(i)][j+1<<level] == 
                bt[right(i)][j+1<<level] == true || continue
            # child cost of current shift of current node
            cc = bc[left(i)][j] + bc[right(i)][j]      
            # child cost of circshift of current node               
            scc = bc[left(i)][j+1<<level] + bc[right(i)][j+1<<level]  
            mincost = min(cc, scc, bc[i][j])
            # mincost=parent, delete subtrees of both cc & scc                    
            if mincost == bc[i][j]          
                delete_subtree!(bt, left(i), j)
                delete_subtree!(bt, right(i), j)
                delete_subtree!(bt, left(i), j+1<<level)
                delete_subtree!(bt, right(i), j+1<<level)
            # mincost=cc, delete subtrees of scc
            elseif mincost == cc                                             
                bc[i][j] = mincost
                delete_subtree!(bt, left(i), j+1<<level)
                delete_subtree!(bt, right(i), j+1<<level)
            # mincost=scc, delete subtrees of cc
            else                                                                    
                bc[i][j] = mincost
                delete_subtree!(bt, left(i), j)
                delete_subtree!(bt, right(i), j)
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
    if (left(i)) < length(bt)
        if bt[left(i)]
            delete_subtree!(bt, left(i))
        end
        if bt[right(i)]
            delete_subtree!(bt, right(i))
        end
    end
    return nothing
end

function delete_subtree!(bt::AbstractVector{BitVector}, i::Integer, j::Integer)
    @assert 1 <= i <= length(bt)
    level = floor(Int, log2(i))
    bt[i][j] = false
    if (left(i)) < length(bt)          # current node can contain subtrees
        if bt[left(i)][j]              # left child of current shift
            delete_subtree!(bt, left(i), j)
        end
        if bt[right(i)][j]            # right child of current shift
            delete_subtree!(bt, right(i), j)
        end
        if bt[left(i)][j+1<<level]     # left child of added shift
            delete_subtree!(bt, left(i), j+1<<level)
        end
        if bt[right(i)][j+1<<level]   # right child of added shift
            delete_subtree!(bt, right(i), j+1<<level)
        end
    end
    return nothing
end


## BEST BASIS TREES
"""
    bestbasistree(X[, method])

Extension to the best basis tree function from Wavelets.jl. Given a set of 
decomposed signals, returns different types of best basis trees based on the 
methods specified. Available methods are the joint best basis (`JBB()`), least 
statistically dependent basis (`LSDB()`), and individual regular best basis 
(`BB()`).
"""
function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3},                
        method::LSDB) where T<:AbstractFloat
    
    L = method.redundant ? floor(Int, log2(size(X,1))) : size(X,2)-1
    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, L)
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3}, 
        method::JBB) where T<:AbstractFloat

    L = method.redundant ? floor(Int, log2(size(X,1))) : size(X,2)-1
    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, L)
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,3},                
        method::BB) where T<:AbstractFloat
    
    L = method.redundant ? floor(Int, log2(size(X,1))) : size(X,2)-1
    besttree = falses(1<<L-1, size(X,3))
    @inbounds begin
        for i in axes(besttree,2)
            costs = tree_costs(X[:,:,i], method)
            besttree[:,i] = bestbasis_treeselection(costs, L)
        end
    end
    return besttree
end

function Wavelets.Threshold.bestbasistree(X::AbstractArray{T,2}, 
        method::BB) where T<:AbstractFloat

    L = method.redundant ? floor(Int, log2(size(X,1))) : size(X,2)-1
    costs = tree_costs(X, method)
    besttree = bestbasis_treeselection(costs, L)
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


## BEST BASIS EXPANSION COEFFICIENTS
"""
    bestbasiscoef(X, tree)
    bestbasiscoef(X, wt, tree)

Returns the expansion coefficients based on the given tree(s) and wavelet packet
decomposition (WPD) expansion coefficients. If the WPD expansion coefficients 
were not provided, the expansion coefficients can be obtained by providing the
signals and wavelet.
"""
function bestbasiscoef(X::AbstractArray{T, 2}, tree::BitVector) where 
        T<:AbstractFloat
    L = size(X,2)-1
    @assert length(tree) == 1<<L-1
    
    (n,L) = size(X)
    return reshape(bestbasiscoef(reshape(X, (n,L,1)), tree), :)
end

function bestbasiscoef(X::AbstractArray{T,3}, tree::BitVector) where 
        T<:AbstractFloat

    L = size(X,2)-1
    @assert length(tree) == 1<<L-1
    
    n = size(X, 1)
    leaf = getleaf(tree)
    y = X[:,1,:]
    for (i, val) in enumerate(leaf)
        if val
            # if node is selected, use coefficients of the children of the node
            lvl = floor(Integer, log2(i))   # counting of lvl starts from 0
            node = i - 2^lvl                # counting of node starts from 0
            n₀ = nodelength(n, lvl)
            rng = (node * n₀ + 1):((node + 1) * n₀)
            @inbounds y[rng,:] = X[rng, lvl+1, :]
        end
    end
    return y
end

function bestbasiscoef(X::AbstractArray{T,3}, tree::BitArray{2}) where 
        T<:AbstractFloat
    @assert size(X,3) == size(tree,2)
    @assert size(X,1) == size(tree,1) + 1
    
    (n,_,N) = size(X)
    y = Array{T,2}(undef, (n,N))
    for i in axes(X,3)
        @inbounds y[:,i] = bestbasiscoef(X[:,:,i], tree[:,i])
    end
    return y
end

function bestbasiscoef(X::AbstractVector{T}, wt::DiscreteWavelet, 
        tree::BitVector) where T<:AbstractFloat
    
    @assert isvalidtree(X, tree)
    return wpt(X, wt, tree)
end

function bestbasiscoef(X::AbstractArray{T,2}, wt::DiscreteWavelet, 
        tree::BitVector) where T<:AbstractFloat
    @assert isvalidtree(X[:,1], tree)  
    (n, N) = size(X)
    y = Array{T,2}(undef, (n, N))
    for i in axes(y,2)
        @inbounds y[:,i] = wpt(X[:,i], wt, tree)
    end
    return y
end

function bestbasiscoef(X::AbstractArray{T,2}, wt::DiscreteWavelet, 
        tree::BitArray{2}) where T<:AbstractFloat
    @assert size(X,2) == size(tree,2)
    @assert size(X,1) == size(tree,1) + 1    

    (n, N) = size(X)
    y = Array{T,2}(undef, (n, N))
    for i in axes(y,2)
        @inbounds y[:,i] = wpt(X[:,i], wt, tree[:,i])
    end
    return y
end

end # end of module