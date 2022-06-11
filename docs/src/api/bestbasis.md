# Best Basis

```@index
Modules = [BestBasis]
```

## Cost functions and computations
### Public API
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
BestBasis.tree_costs
```

### Private API
```@docs
BestBasis.coefcost
```

# Best Basis Tree Selection
### Private API
```@docs
BestBasis.bestbasis_treeselection
BestBasis.delete_subtree!
```

## Best basis computation
### Public API
```@docs
BestBasis.BestBasisType
BestBasis.LSDB
BestBasis.JBB
BestBasis.BB
Wavelets.Threshold.bestbasistree
BestBasis.bestbasistreeall
```
