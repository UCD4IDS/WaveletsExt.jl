# Wavelet Packet Decomposition

```@index
Modules = [DWT]
```

## Public API
### Transforms on 1 Signal
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

### Transforms on Multiple Signals
```@docs
DWT.dwtall
DWT.idwtall
DWT.wptall
DWT.iwptall
DWT.wpdall
DWT.iwpdall
```

## Private API
### Single Step Transforms
```@docs
DWT.dwt_step
DWT.dwt_step!
DWT.idwt_step
DWT.idwt_step!
```