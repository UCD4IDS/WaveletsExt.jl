# Autocorrelation Wavelet Transform

```@index
Modules = [ACWT]
```

## Public API
### Transforms on 1 Signal
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

### Transforms on Multiple Signals
!!! note
    The following functions currently only support 1D-signals. Transforms on multiple 2D-signals are not yet supported.
```@docs
ACWT.acdwtall
ACWT.iacdwtall
ACWT.acwptall
ACWT.iacwptall
ACWT.acwpdall
ACWT.iacwpdall
```

## Private API
### Utilities
```@docs
ACWT.autocorr
ACWT.pfilter
ACWT.qfilter
ACWT.make_acqmfpair
ACWT.make_acreverseqmfpair
```

### Single Step Transforms
```@docs
ACWT.acdwt_step
ACWT.acdwt_step!
ACWT.iacdwt_step
ACWT.iacdwt_step!
```