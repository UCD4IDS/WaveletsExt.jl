```@raw html
<div style="width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;
        border-color:#000;border-radius:10px;text-align:center;background-color:#B3D8FF;
        color:#000">
    <h3 style="color: black;">Star us on GitHub!</h3>
    <a class="github-button" href="https://github.com/UCD4IDS/WaveletsExt.jl.git" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star UCD4IDS/WaveletsExt.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

# WaveletsExt.jl

This package is a [Julia](https://github.com/JuliaLang/julia) extension package to [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl) (WaveletsExt is short for Wavelets Extension). It contains additional functionalities that complement Wavelets.jl, which include multiple best basis algorithms, denoising methods, [Local Discriminant Basis (LDB)](https://www.math.ucdavis.edu/~saito/publications/saito_ldb_jmiv.pdf), [Stationary Wavelet Transform](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.2662&rep=rep1&type=pdf), and the [Shift Invariant Wavelet Decomposition](https://israelcohen.com/wp-content/uploads/2018/05/ICASSP95.pdf).

## Authors
This package is written and maintained by Zeng Fung Liew and Shozen Dan under the supervision of Professor Naoki Saito at the University of California, Davis.

## Installation
The package is part of the official Julia Registry. It can be install via the Julia REPL.
```julia
(v1.6) pkg> add WaveletsExt
```
or
```julia
julia> using Pkg; Pkg.add("WaveletsExt")
```
## Usage
Load the WaveletsExt module along with Wavelets.jl.
```julia
using Wavelets, WaveletsExt
```

## Wavelet Packet Decomposition
In contrast to Wavelets.jl's `wpt` function, `wpd` outputs expansion coefficients of all levels of a given signal. Each column represents a level in the decomposition tree.
```julia
y = wpd(x, wavelet(WT.db4))
```

## Stationary Wavelet Transform
The redundant and non-orthogonal transform by Nason-Silverman can be implemented using either [`sdwt`](@ref WaveletsExt.SWT.sdwt) (for stationary discrete wavelet transform) or [`swpd`](@ref WaveletsExt.SWT.swpd) (for stationary wavelet packet decomposition). Similarly, the reconstruction of signals can be computed using [`isdwt`](@ref WaveletsExt.SWT.isdwt) and [`iswpt`](@ref WaveletsExt.SWT.iswpt).
```julia
# stationary discrete wavelet transform
y = sdwt(x, wavelet(WT.db4))
z = isdwt(y, wavelet(WT.db4))

# stationary wavelet packet decomposition
y = swpd(x, wavelet(WT.db4))
z = iswpt(y, wavelet(WT.db4))
```

## Best Basis
In addition to the best basis algorithm by M.V. Wickerhauser implemented in Wavelets.jl, WaveletsExt.jl contains the implementation of the Joint Best Basis (JBB) by Wickerhauser an the [Least Statistically-Dependent Basis (LSDB)](https://www.math.ucdavis.edu/~saito/courses/ACHA.suppl/lsdb-pr-journal.pdf) by N. Saito.
```julia
y = cat([wpd(x[:,i], wt) for i in N]..., dims=3)    # x has size (2^L, N)

# individual best basis trees
bbt = bestbasistree(y, BB())
# joint best basis
bbt = bestbasistree(y, JBB())
# least statistically dependent basis
bbt = bestbasistree(y, LSDB())
```
Given a `BitVector` representing a best basis tree, one can obtain the corresponding expansion coefficients using [`bestbasiscoef`](@ref WaveletsExt.BestBasis.bestbasiscoef).
```julia
coef = bestbasiscoef(y, bbt)
```
For more information on the different wavelet transforms and best basis algorithms, please refer to its [manual](@ref transforms_manual).

## Signal Denoising
WaveletsExt.jl includes additional signal denoising and thresholding methods that complement those written in Wavelets.jl. One can denoise a signal as using the [`denoise`](@ref WaveletsExt.Denoising.denoise) Wavelets.jl extension function as follows:
```julia
x̂ = denoise(y, :wpt, wt, tree=bt)
```
Additionally, for cases where there are multiple signals to be denoised, one can use the [`denoiseall`](@ref WaveletsExt.Denoising.denoiseall) function as below.
```julia
X̂ = denoiseall(Y, :wpt, wt, tree=bt)
```

## Local Discriminant Basis
Local Discriminant Basis (LDB) is a feature extraction method developed by N. Saito and R. Coifman and can be accessed as follows.
```julia
# generate data
X, y = generateclassdata(ClassData(:tri, 5, 5, 5))
wt = wavelet(WT.haar)

# LDB
f = LocalDiscriminantBasis(wt, top_k=5, n_features=5)
Xt = fit_transform(f, X, y)
```
For more information on how to use LDB, please refer to its [manual](@ref ldb_manual).