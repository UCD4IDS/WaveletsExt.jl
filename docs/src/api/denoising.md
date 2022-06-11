# [Denoising](@id denoising_api)

```@index
Modules = [Denoising]
```

## Shrinking Types and Constructors
### Public API
```@docs
Denoising.RelErrorShrink
Denoising.SureShrink
Denoising.SureShrink(::AbstractArray{T}, ::Bool, ::Union{BitVector, Nothing}, ::Wavelets.Threshold.THType) where T<:Number
Wavelets.Threshold.VisuShrink
```

## Threshold Determination and Noise Estimation
### Public API
```@docs
Wavelets.Threshold.noisest
Denoising.relerrorthreshold
```

## Denoising Functions
### Public API
```@docs
Wavelets.Threshold.denoise
Denoising.denoiseall
```

## Helper Functions for Threshold Determination and Noise Estimation
### Private API
```@docs
Denoising.surethreshold
Denoising.orth2relerror
Denoising.findelbow
Denoising.relerrorplot
```