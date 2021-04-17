# Denoising Examples

## Denoising a single signal
```@example
using Wavelets, WaveletsExt, Random

# define function and wavelet
x = testfunction(256, "HeaviSine") + randn(256, 0.5)
wt = wavelet(WT.db4)

# best basis tree
bt = bestbasistree(wpd(x, wt), BB())
y = bestbasiscoef(x, wt, bt)

# denoise
x̂ = denoise(y, :wpt, wt, tree=bt)
```

## Denoising a group of signals
```@example
using Wavelets, WaveletsExt, Random

# define function and wavelet
x = testfunction(256, "HeaviSine")
X = generatesignals(x, 10, 2, true, 0.5)
wt = wavelet(WT.db4)

# decomposition
coef = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(coef, JBB())
Y = bestbasiscoef(coef, bt)

# denoise
X̂ = denoiseall(Y, :wpt, wt, tree=bt)
```