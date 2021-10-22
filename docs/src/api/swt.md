# Stationary Wavelet Transform

```@index
Modules = [SWT]
```

## Public API
!!! note
    All SWT functions currently only support 1D-signals. Transforms on multiple 2D-signals are not yet supported.
### Transforms on 1 Signal
```@docs
SWT.sdwt
SWT.sdwt!
SWT.isdwt
SWT.isdwt!
SWT.swpt
SWT.swpt!
SWT.iswpt
SWT.iswpt!
SWT.swpd
SWT.swpd!
SWT.iswpd
SWT.iswpd!
```

### Transforms on Multiple Signals
```@docs
SWT.sdwtall
SWT.isdwtall
SWT.swptall
SWT.iswptall
SWT.swpdall
SWT.iswpdall
```

## Private API
### Single Step Transforms
```@docs
SWT.sdwt_step
SWT.sdwt_step!
SWT.isdwt_step
SWT.isdwt_step!
```