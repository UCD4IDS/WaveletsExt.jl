# Best Basis

```@index
Modules = [BestBasis]
```

## Public API
### Cost functions and computations
```@docs
BestBasis.CostFunction
BestBasis.LSDBCost
BestBasis.JBBCost
BestBasis.BBCost
BestBasis.LoglpCost
BestBasis.NormCost
BestBasis.DifferentialEntropyCost
BestBasis.ShannonEntropyCost
BestBasis.LogEnergyEntropyCost
BestBasis.coefcost
BestBasis.tree_costs
BestBasis.tree_costs(::AbstractMatrix{T}, ::AbstractVector{BitVector}, ::SIBB) where T<:Number
```

### Best basis computation
```@docs
BestBasis.BestBasisType
BestBasis.LSDB
BestBasis.JBB
BestBasis.BB
BestBasis.SIBB
Wavelets.Threshold.bestbasistree
Wavelets.Threshold.bestbasistree(::AbstractMatrix{T}, ::Integer, ::SIBB) where T<:Number
BestBasis.bestbasistreeall
```

# Private API
```@docs
BestBasis.bestbasis_treeselection
BestBasis.bestbasis_treeselection(::AbstractVector{Tc}, ::AbstractVector{Tt}) where {Tc<:AbstractVector{<:Union{Number,Nothing}}, Tt<:BitVector}
BestBasis.delete_subtree!

```