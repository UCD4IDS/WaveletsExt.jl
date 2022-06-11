# Autocorrelation Wavelet Transform

```@index
Modules = [ACWT]
```

## Transforms on 1 Signal
### Public API
```@docs
ACWT.acdwt
ACWT.acdwt!
ACWT.iacdwt
ACWT.iacdwt!
ACWT.acwpt
ACWT.acwpt!
ACWT.iacwpt
ACWT.iacwpt!
ACWT.acwpd
ACWT.acwpd!
ACWT.iacwpd
ACWT.iacwpd!
```

## Transforms on Multiple Signals
### Public API
```@docs
ACWT.acdwtall
ACWT.iacdwtall
ACWT.acwptall
ACWT.iacwptall
ACWT.acwpdall
ACWT.iacwpdall
```

## Utilities
### Private API
```@docs
ACWT.autocorr
ACWT.pfilter
ACWT.qfilter
ACWT.make_acqmfpair
ACWT.make_acreverseqmfpair
```

## Single Step Transforms
### Private API
```@docs
ACWT.acdwt_step
ACWT.acdwt_step!
ACWT.iacdwt_step
ACWT.iacdwt_step!
```