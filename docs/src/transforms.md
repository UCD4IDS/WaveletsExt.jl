# Wavelet Transforms

## Wavelet Packet Decomposition

### Example
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
wt = wavelet(WT.db4)

# decomposition
y = wpd(x, wt)
```

## Stationary Discrete Wavelet Transform

### Example
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
wt = wavelet(WT.db4)

# transform
y = sdwt(x, wt)
```

## Stationary Wavelet Packet Decomposition

### Example
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
wt = wavelet(WT.db4)

# decomposition
y = swpd(x, wt)
```

## Shift-Invariant Wavelet Packet Decomposition

### Example
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
wt = wavelet(WT.db4)

# decomposition
y = siwpd(x, wt, maxtransformlevels(x),)
```