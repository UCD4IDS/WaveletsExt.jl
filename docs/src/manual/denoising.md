# Denoising Examples
For more information and examples on wavelet denoising, visit [Wavelets Denoising Experiment repository under UCD4IDS](https://github.com/UCD4IDS/WaveletsDenoisingExperiment) for a step-by-step tutorial in a Pluto notebook.

## Denoising a single signal
```@example
using Wavelets, WaveletsExt, Random

# define function and wavelet
x = generatesignals(:heavysine, 8) + 0.5*randn(256)
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
x = generatesignals(:heavysine, 8)
X = duplicatesignals(x, 10, 2, true, 0.5)
wt = wavelet(WT.db4)

# decomposition
coef = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(coef, JBB())
Y = bestbasiscoef(coef, bt)

# denoise
X̂ = denoiseall(Y, :wpt, wt, tree=bt)
```