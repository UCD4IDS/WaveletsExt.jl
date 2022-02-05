# [Fast Numerical Algorithms using Wavelet Transforms](@id wavemult_manual)
This demonstration is based on the paper *[Fast wavelet transforms and numerical algorithms](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.3160440202)* by G. Beylkin, R. Coifman, and V. Rokhlin and the lecture notes *[Wavelets and Fast Numerical ALgorithms](https://arxiv.org/abs/comp-gas/9304004)* by G. Beylkin.

Assume we have a matrix ``M \in \mathbb{R}^{n \times n}``, and a vector ``x \in \mathbb{R}^n``, there are two ways to compute ``y = Mx``:  

1) Use the regular matrix multiplication, which takes ``O(n^2)`` time.
2) Transform both ``M`` and ``x`` to their standard/nonstandard forms ``\tilde M`` and ``\tilde x`` respectively. Compute the multiplication of ``\tilde y = \tilde M \tilde x``, then find the inverse of ``\tilde y``.If the matrix ``\tilde M`` is sparse, this can take ``O(n)`` time.

## Examples and Comparisons
In our following examples, we compare the methods described in the previous section. We focus on two important metrics: time taken and accuracy of the algorithm.

We start off by setting up the matrix `M`, the vector `x`, and the wavelet `wt`. As this algorithm is more efficient when ``n`` is large, we set ``n=2048``.
```@example wavemult
using Wavelets, WaveletsExt, Random, LinearAlgebra, BenchmarkTools

# Construct matrix M, vector x, and wavelet wt
function data_setup(n::Integer, random_state = 1234)
    M = zeros(n,n)
    for i in axes(M,1)
        for j in axes(M,2)
            M[i,j] = i==j ? 0 : 1/abs(i-j)
        end
    end

    Random.seed!(random_state)
    x = randn(n)

    return M, x
end

M, x = data_setup(2048)
wt = wavelet(WT.haar)
nothing     # hide
```

The following code block computes the regular matrix multiplication.
```@example wavemult
y₀ = M * x
nothing     # hide
```

The following code block computes the nonstandard wavelet multiplication. Note that we will only be running 4 levels of wavelet transform, as running too many levels result in large computation time, without much improvement in accuracy.
```@example wavemult
L = 4
NM = mat2sparseform_nonstd(M, wt, L)
y₁ = nonstd_wavemult(NM, x, wt, L)
nothing     # hide
```

The following code block computes the standard wavelet multiplication. Once again, we will only be running 4 levels of wavelet transform here.
```@example wavemult
SM = mat2sparseform_std(M, wt, L)
y₂ = std_wavemult(SM, x, wt, L)
nothing     # hide
```

!!! tip "Performance Tip"
    Running more levels of transform (i.e. setting large values of `L`) does not necessarily result in more accurate approximations. Setting `L` to be a reasonably large value (e.g. 3 or 4) not only reduces computational time, but could also produce better results.

We assume that the regular matrix multiplication produces the true results (bar some negligible machine errors), and we compare our results from the nonstandard and standard wavelet multiplications based on their relative errors (``\frac{||\hat y - y_0||_2}{||y_0||_2}``).
```@repl wavemult
relativenorm(y₁, y₀)       # Comparing nonstandard wavelet multiplication with true value
relativenorm(y₂, y₀)       # Comparing standard wavelet multiplication with true value
```

Lastly, we compare the computation time for each algorithm.
```@repl wavemult
@benchmark $M * $x                          # Benchmark regular matrix multiplication
@benchmark nonstd_wavemult($NM, $x, $wt, L) # Benchmark nonstandard wavelet multiplication
@benchmark std_wavemult($SM, $x, $wt, L)    # Benchmark standard wavelet multiplication
```

As we can see above, the nonstandard and standard wavelet multiplication methods are significantly faster than the regular method and provides reasonable approximations to the true values.

!!! tip "Performance Tips"
    This method should only be used when the dimensions of ``M`` and ``x`` are large. Additionally, this would be more useful when one aims to compute ``Mx_0, Mx_1, \ldots, Mx_k`` where ``M`` remains constant throughout the ``k`` computations. In this case, it is more efficient to use 
    ```julia
    NM = mat2sparseform_nonstd(M, wt)
    y = nonstd_wavemult(NM, x, wt)
    ```
    instead of
    ```julia
    y = nonstd_wavemult(M, x, wt)
    ```