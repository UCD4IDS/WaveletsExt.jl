# WaveletsExt.jl

| Docs | Build | Test |
|------|-------|------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://UCD4IDS.github.io/WaveletsExt.jl/stable) | [![CI](https://github.com/UCD4IDS/WaveletsExt.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/UCD4IDS/WaveletsExt.jl/actions) | [![codecov](https://codecov.io/gh/UCD4IDS/WaveletsExt.jl/branch/master/graph/badge.svg?token=U3EOscAvPE)](https://codecov.io/gh/UCD4IDS/WaveletsExt.jl) |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://UCD4IDS.github.io/WaveletsExt.jl/dev) | | |



This package is a [Julia](https://github.com/JuliaLang/julia) extension package to [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl) (WaveletsExt is short for Wavelets Extension). It contains additional functionalities that complement Wavelets.jl, which include multiple best basis algorithms, denoising methods, [Local Discriminant Basis (LDB)](https://www.math.ucdavis.edu/~saito/publications/saito_ldb_jmiv.pdf), [Stationary Wavelet Transform](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.2662&rep=rep1&type=pdf), [Autocorrelation Wavelet Transform (ACWT)](https://www.math.ucdavis.edu/~saito/publications/saito_minframe.pdf), and the [Shift Invariant Wavelet Decomposition](https://israelcohen.com/wp-content/uploads/2018/05/ICASSP95.pdf).

## Authors
This package is written and maintained by Zeng Fung Liew and Shozen Dan under the supervision of Professor Naoki Saito at the University of California, Davis.

## Installation
The package is part of the official Julia Registry. It can be install via the Julia REPL.
```julia
(@1.6) pkg> add WaveletsExt
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
The redundant and non-orthogonal transform by Nason-Silverman can be implemented using either `sdwt` (for stationary discrete wavelet transform) or `iswpd` (for stationary wavelet packet decomposition). Similarly, the reconstruction of signals can be computed using `isdwt` and `iswpt`.
```julia
# stationary discrete wavelet transform
y = sdwt(x, wavelet(WT.db4))
z = isdwt(y, wavelet(WT.db4))

# stationary wavelet packet decomposition
y = swpd(x, wavelet(WT.db4))
z = iswpt(y, wavelet(WT.db4))
```

## Autocorrelation Wavelet Transform
The [autocorrelation wavelet transform (ACWT)](https://www.math.ucdavis.edu/~saito/publications/saito_minframe.pdf) is a special case of the stationary wavelet transform. Some desirable properties of ACWT are symmetry without losing vanishing moments, edge detection/characterization capabilities, and shift invariance. To transform a signal using AC wavelets, use `acwt` (discreate AC wavelet transform) or `acwpt` (a.c. packet transform). `acwt` can also handle 2D signals, which is useful in applications such as image denoising or compression. The reconstruction of signals can be performed using `iacwt` and `iacwpt`.
```julia
# autocorrelation discrete wavelet transform
y = acwt(x, wavelet(WT.db4))
z = iacwt(y)

# autocorrelation wavelet packet transform
tree = maketree(x, :full)
y = acwpt(x, wavelet(WT.db4))
z = iacwpt(y, tree)
```

## Best Basis
In addition to the best basis algorithm by M.V. Wickerhauser implemented in Wavelets.jl, WaveletsExt.jl contains the implementation of the Joint Best Basis (JBB) by Wickerhauser an the [Least Statistically-Dependent Basis (LSDB)](https://www.math.ucdavis.edu/~saito/courses/ACHA.suppl/lsdb-pr-journal.pdf) by Saito.
```julia
y = cat([wpd(x[:,i], wt) for i in N]..., dims=3)    # x has size (2^L, N)

# individual best basis trees
bbt = bestbasistree(y, BB())
# joint best basis
bbt = bestbasistree(y, JBB())
# least statistically dependent basis
bbt = bestbasistree(y, LSDB())
```
Given a `BitVector` representing a best basis tree, one can obtain the corresponding expansion coefficients using `bestbasiscoef`.
```julia
coef = bestbasiscoef(y, bbt)
```

## Local Discriminant Basis
Local Discriminant Basis (LDB) is a feature extraction method developed by Naoki Saito.
```julia
X, y = generateclassdata(ClassData(:tri, 5, 5, 5))
wt = wavelet(WT.haar)

f = LocalDiscriminantBasis(wt, top_k=5, n_features=5)
Xt = fit_transform(f, X, y)
```

## TODO(By next patch release):
* Improve webpage "Manual" documentation 
* Bug fix on cases where `f.n_features` is changed after `change_nfeatures` function is ran but resulting output is not saved.
* More checking on class attributes for `transform` function in LDB.jl.

## TODO(long term):
* Inverse Transforms for Shift-Invariant WPT
* Improve API for LDB by utilizing `fit`, `fit_transform`, `transform` functions.
* Better documentation.