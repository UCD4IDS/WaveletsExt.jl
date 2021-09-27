# [Signal Denoising](@id denoising_manual)
Wavelet denoising is an important step in signal analysis as it helps remove unnecessary high frequency noise while maintaining the most important features of the signal. Intuitively, signal denoising comes in the following simple steps:
1. Decompose a signal or a group of signals. One can choose to decompose signals into its best basis tree for more optimal results.
2. Find a suitable threshold value. There are many ways to do so, with VisuShrink (D. Donoho, I. Johnstone) being one of the most popular approaches. The VisuShrink implementation in Wavelets.jl, along with the [RelErrorShrink](@ref WaveletsExt.Denoising.RelErrorShrink) and the [SureShrink](@ref WaveletsExt.Denoising.SureShrink) implementations in WaveletsExt.jl give users more threshold selection options.
3. Threshold the wavelet coefficients. There are various thresholding methods implemented in Wavelets.jl for this purpose, with Hard and Soft thresholding being the usual go-to method due to its simplistic approach.
4. Reconstruct the original signals using the thresholded coefficients.

For more information and examples on wavelet denoising using WaveletsExt.jl, visit [Wavelets Denoising Experiment repository under UCD4IDS](https://github.com/UCD4IDS/WaveletsDenoisingExperiment) for a step-by-step tutorial in a Pluto notebook. The following is a simple guide on denoisng using WaveletsExt.jl.

## Denoising a single signal
To denoise a single signal, one can use the `denoise` function from WaveletsExt.jl as shown below. Note the following key parameters:
- `x`: Input signal.
- `inputtype`: Type of input. One can input an original signal `:sig`, or first transform the signal and type in one of `:dwt`, `:wpt`, `:sdwt`, `:swpd`, `:acwt`, and `:acwpt`.
- `wt`: Transform wavelet.
- `L`: Number of decomposition levels. Necessary for input types `:sig`, `:dwt`, and `:sdwt`.
- `tree`: Decomposition tree of the signals. Necessary for input types `:wpt` and `:swpd`.
- `dnt`: Denoise type. One should input either of VisuShrink, RelErrorShrink, or SureShrink.
- `estnoise`: Noise estimation. Can be a function or a value.
- `smooth`: Smoothing method used.
For more detailed information, visit the [denoising API](@ref denoising_api) page.

```@example
using Wavelets, WaveletsExt, Random, Plots

# define function and wavelet
x₀ = generatesignals(:heavysine, 8)
x = x₀ + 0.8*randn(256)
wt = wavelet(WT.db4)

# best basis tree
bt = bestbasistree(wpd(x, wt), BB())
y = bestbasiscoef(x, wt, bt)

# denoise
x̂ = denoise(y, :wpt, wt, tree=bt)

# plot results
nothing # hide
plot([x₀ x x̂], title="Denoising Example", label=["original" "noisy" "denoised"],
     lw=[3 1 2], lc=[:black :grey :red])
```

## Denoising a group of signals
Similar to the `denoise` function we saw previously, for denoising a group of signals, one can use the `denoiseall` function. The parameters used are the same, with the following addition:
- `bestTH`: Method to determine the best threshold value for a group of signals. One can choose each signal's individual best threshold value, or use a function such as `mean` or `median` to generalize an overall best threshold value.

For more detailed information, visit the [denoising API](@ref denoising_api) page.
```@example
using Wavelets, WaveletsExt, Random, Plots

# define function and wavelet
x = generatesignals(:heavysine, 8)
X₀ = duplicatesignals(x, 6, 2, false)
X = duplicatesignals(x, 6, 2, true, 0.8)
wt = wavelet(WT.db4)

# decomposition
coef = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(coef, JBB())
Y = bestbasiscoef(coef, bt)

# denoise
X̂ = denoiseall(Y, :wpt, wt, tree=bt)

# plot results
nothing # hide
wiggle(X₀, sc=0.7, FaceColor=:white, ZDir=:reverse)
wiggle!(X, sc=0.7, EdgeColor=:grey, FaceColor=:white, ZDir=:reverse)
wiggle!(X̂, sc=0.7, EdgeColor=:red, FaceColor=:white, ZDir=:reverse)
plot!(title="Group Denoising Example")
```