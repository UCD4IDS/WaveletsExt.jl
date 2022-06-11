# Utils

```@index
Modules = [Utils]
```

## Useful wavelet/signal utilities
### Public API
```@docs
Wavelets.Util.maxtransformlevels
Utils.getbasiscoef
Utils.getbasiscoefall
Utils.getrowrange
Utils.getcolrange
Utils.nodelength
Utils.coarsestscalingrange
Utils.finestdetailrange
```

## Tree traversing functions
### Public API
```@docs
Wavelets.Util.isvalidtree(::AbstractMatrix,::BitVector)
Wavelets.Util.maketree
Utils.getchildindex
Utils.getparentindex
Utils.getleaf
Utils.getdepth
Utils.gettreelength
```

## Metrics
### Public API
```@docs
Utils.relativenorm
Utils.psnr
Utils.snr
Utils.ssim
```

## Dataset generation
### Public API
```@docs
Utils.ClassData
Utils.duplicatesignals
Utils.generatesignals
Utils.generateclassdata
```

## Miscellaneous
### Private API
```@docs
Utils.main2depthshift
```