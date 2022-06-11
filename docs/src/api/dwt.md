# Standard Wavelet Transforms

```@index
Modules = [DWT]
```

## Transforms on 1 Signal
### Public API
```@docs
Wavelets.Transforms.wpt
Wavelets.Transforms.wpt(::AbstractArray{T,2}, ::OrthoFilter, ::Integer; ::Bool) where T<:Number
Wavelets.Transforms.wpt!
Wavelets.Transforms.wpt!(::AbstractArray{T,2}, ::AbstractArray{T,2}, ::OrthoFilter, ::Integer; ::Bool) where T<:Number
Wavelets.Transforms.iwpt
Wavelets.Transforms.iwpt(::AbstractArray{T,2}, ::OrthoFilter, ::Integer; ::Bool) where T<:Number
Wavelets.Transforms.iwpt!
Wavelets.Transforms.iwpt!(::AbstractArray{T,2}, ::AbstractArray{T,2}, ::OrthoFilter, ::Integer; ::Bool) where T<:Number
DWT.wpd
DWT.wpd!
DWT.iwpd
DWT.iwpd!
```

## Transforms on Multiple Signals
### Public API
```@docs
DWT.dwtall
DWT.idwtall
DWT.wptall
DWT.iwptall
DWT.wpdall
DWT.iwpdall
```

## Single Step Transforms
### Private API
```@docs
DWT.dwt_step
DWT.dwt_step!
DWT.idwt_step
DWT.idwt_step!
```