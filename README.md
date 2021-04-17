# WaveletsExt.jl

[![CI](https://github.com/zengfung/WaveletsExt.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/zengfung/WaveletsExt.jl/actions)
[![codecov](https://codecov.io/gh/zengfung/WaveletsExt.jl/branch/master/graph/badge.svg?token=3J520FN4J2)](https://codecov.io/gh/zengfung/WaveletsExt.jl)

This package is a [Julia](https://github.com/JuliaLang/julia) extension package to [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl) (WaveletsExt is short for Wavelets Extension). It contains additional functionalities that complement Wavelets.jl, which include multiple best basis algorithms, denoising methods, [Local Discriminant Basis (LDB)](https://www.math.ucdavis.edu/~saito/publications/saito_ldb_jmiv.pdf), [Stationary Wavelet Transform](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.2662&rep=rep1&type=pdf), [Autocorrelation Wavelet Transform (ACWT)](https://www.math.ucdavis.edu/~saito/publications/saito_minframe.pdf), and the [Shift Invariant Wavelet Decomposition](https://israelcohen.com/wp-content/uploads/2018/05/ICASSP95.pdf).

## Authors
This package is written and maintained by Zeng Fung Liew and Shozen Dan under the supervision of Professor Naoki Saito at the University of California, Davis.

## Installation
The package is part of the official Julia Registry. It can be install via the Julia REPL.
```
(@1.x) pkg> add WaveletsExt
```
or
```
julia> using Pkg; Pkg.add("WaveletsExt")
```
## Usage
Load the WaveletsExt module along with Wavelets.jl.
```
using Wavelets, WaveletsExt
```

## Wavelet Packet Decomposition
In contrast to Wavelets.jl's `wpt` function, `wpd` outputs expansion coefficients of all levels of a given signal. Each column represents a level in the decomposition tree.
```
y = wpd(x, wavelet(WT.db4))
```

## Stationary Wavelet Transform
The redundant and non-orthogonal transform by Nason-Silverman can be implemented using either `sdwt` (for stationary discrete wavelet transform) or `iswpd` (for stationary wavelet packet decomposition). Similarly, the reconstruction of signals can be computed using `isdwt` and `iswpt`.
```
# stationary discrete wavelet transform
y = sdwt(x, wavelet(WT.db4))
z = isdwt(y, wavelet(WT.db4))

# stationary wavelet packet decomposition
y = swpd(x, wavelet(WT.db4))
z = iswpt(y, wavelet(WT.db4))
```

## Autocorrelation Wavelet Transform
The [autocorrelation wavelet transform (ACWT)](https://www.math.ucdavis.edu/~saito/publications/saito_minframe.pdf) is a special case of the stationary wavelet transform. Some desirable properties of ACWT are symmetry without losing vanishing moments, edge detection/characterization capabilities, and shift invariance. To transform a signal using AC wavelets, use `acwt` (discreate AC wavelet transform) or `acwpt` (a.c. packet transform). `acwt` can also handle 2D signals, which is useful in applications such as image denoising or compression. The reconstruction of signals can be performed using `iacwt` and `iacwpt`.
```
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
```
y = cat([wpd(x[:,i], wt) for i in N]..., dims=3)    # x has size (2^L, N)

# individual best basis trees
bbt = bestbasistree(y, BB())
# joint best basis
bbt = bestbasistree(y, JBB())
# least statistically dependent basis
bbt = bestbasistree(y, LSDB())
```
Given a `BitVector` representing a best basis tree, one can obtain the corresponding expansion coefficients using `bestbasiscoef`.
```
coef = bestbasiscoef(y, bbt)
```

## Local Discriminant Basis
Local Discriminant Basis (LDB) is a feature extraction method developed by Naoki Saito.
```{julia}
coef, y, ldb_tree, power, order = ldb(X, y, wavelet(WT.coif3))
```