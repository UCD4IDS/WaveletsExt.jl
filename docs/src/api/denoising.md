# [Denoising](@id denoising_api)

```@index
Modules = [Denoising]
```

# Public API
## Shrinking Types and Constructors
```@docs
Denoising.RelErrorShrink
Denoising.SureShrink
Denoising.SureShrink(::AbstractArray{T}, ::Bool, ::Union{BitVector, Nothing}, ::Wavelets.Threshold.THType) where T<:Number
Wavelets.Threshold.VisuShrink
```

## Threshold Determination and Noise Estimation
```@docs
Wavelets.Threshold.noisest
Denoising.relerrorthreshold
```

## Denoising Functions
```@docs
Wavelets.Threshold.denoise
Denoising.denoiseall
```

# Private API
## Helper Functions for Threshold Determination and Noise Estimation
```@docs
Denoising.surethreshold
Denoising.orth2relerror
Denoising.findelbow
Denoising.relerrorplot
```