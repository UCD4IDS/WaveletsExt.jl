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
This package is a [Julia](https://github.com/JuliaLang/julia) extension package to
[Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl) (WaveletsExt is short for Wavelets
Extension). It contains additional functionalities that complement Wavelets.jl, namely
- Multi-dimensional wavelet transforms
- Redundant wavelet transforms
    - [Autocorrelation Wavelet Transforms (ACWT)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/1826/1/Wavelets-their-autocorrelation-functions-and-multiresolution-representations-of-signals/10.1117/12.131585.short)
    - [Stationary Wavelet Transforms (SWT)](https://doi.org/10.1007/978-1-4612-2544-7_17)
    - [Shift Invariant Wavelet Transforms (SIWT)](https://doi.org/10.1016/S0165-1684(97)00007-8)
- Best basis algorithms
    - [Joint best basis (JBB)](https://ieeexplore.ieee.org/document/119732)
    - [Least statistically dependent basis (LSDB)](https://ieeexplore.ieee.org/document/750958)
- Denoising methods
    - [Relative Error Shrink (RelErrorShrink)](https://ieeexplore.ieee.org/document/7752982)
    - [Stein Unbiased Risk Estimator Shrink (SUREShrink)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1995.10476626)
- Wavelet transform based feature extraction techniques
    - [Local Discriminant Basis (LDB)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2303/1/Local-discriminant-bases/10.1117/12.188763.short).

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

## References
[1] Coifman, R.R., Wickerhauser, M.V. (1992). *Entropy-based algorithms for best basis
selection*. DOI: [10.1109/18.119732](https://ieeexplore.ieee.org/document/119732)

[2] Saito, N. (1998). *The least statistically-dependent basis and its applications*. DOI:
[10.1109/ACSSC.1998.750958](https://ieeexplore.ieee.org/document/750958)

[3] Beylkin, G., Saito, N. (1992). *Wavelets, their autocorrelation functions, and
multiresolution representations of signals*. DOI:
[10.1117/12.131585](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/1826/1/Wavelets-their-autocorrelation-functions-and-multiresolution-representations-of-signals/10.1117/12.131585.short)

[4] Nason, G.P., Silverman, B.W. (1995) *The Stationary Wavelet Transform and some
Statistical Applications*. DOI:
[10.1007/978-1-4612-2544-7_17](https://doi.org/10.1007/978-1-4612-2544-7_17)

[5] Donoho, D.L., Johnstone, I.M. (1995). *Adapting to Unknown Smoothness via Wavelet
Shrinkage*. DOI:
[10.1080/01621459.1995.10476626](https://www.tandfonline.com/doi/abs/10.1080/01621459.1995.10476626)  

[6] Saito, N., Coifman, R.R. (1994). *Local Discriminant Basis*. DOI:
[10.1117/12.188763](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2303/1/Local-discriminant-bases/10.1117/12.188763.short)  

[7] Saito, N., Coifman, R.R. (1995). *Local discriminant basis and their applications*. DOI:
[10.1007/BF01250288](https://doi.org/10.1007/BF01250288)  

[8] Saito, N., Marchand, B. (2012). *Earth Mover's Distance-Based Local Discriminant Basis*.
DOI: [10.1007/978-1-4614-4145-8_12](https://doi.org/10.1007/978-1-4614-4145-8_12)  

[9] Cohen, I., Raz, S., Malah, D. (1997). *Orthonormal shift-invariant wavelet packet
decomposition and representation*. DOI:
[10.1016/S0165-1684(97)00007-8](https://doi.org/10.1016/S0165-1684(97)00007-8)  

[10] Irion, J., Saito, N. (2017). *Efficient Approximation and Denoising of Graph Signals
Using the Multiscale Basis Dictionaries*. DOI: [10.1109/TSIPN.2016.2632039](https://ieeexplore.ieee.org/document/7752982)
```