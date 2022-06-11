# Local Discriminant Basis

```@index
Modules = [LDB]
```

## Energy Maps
### Public API
```@docs
LDB.EnergyMap
LDB.TimeFrequency
LDB.ProbabilityDensity
LDB.Signatures
LDB.energy_map
```

## Discriminant Measures
### Public API
```@docs
LDB.DiscriminantMeasure
LDB.ProbabilityDensityDM
LDB.SignaturesDM
LDB.AsymmetricRelativeEntropy
LDB.SymmetricRelativeEntropy
LDB.HellingerDistance
LDB.LpDistance
LDB.EarthMoverDistance
LDB.discriminant_measure
```

### Private API
```@docs
LDB.pairwise_discriminant_measure
```

## Computation of Discriminant Powers
### Public API
```@docs
LDB.DiscriminantPower
LDB.BasisDiscriminantMeasure
LDB.FishersClassSeparability
LDB.RobustFishersClassSeparability
LDB.discriminant_power
```

## Feature Extraction and Transformation
### Public API
```@docs
LDB.LocalDiscriminantBasis
LDB.fit!
LDB.fitdec!
LDB.fit_transform
LDB.transform
LDB.inverse_transform
LDB.change_nfeatures
```
