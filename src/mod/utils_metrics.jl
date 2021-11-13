# ----- Computational metrics -----
# Relative norm between 2 vectors
"""
    relativenorm(x, x₀[, p]) where T<:Number

Returns the relative norm of base p between original signal x₀ and noisy signal
x.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Signal with noise.
- `x₀::AbstractArray{T} where T<:Number`: Reference signal.
- `p::Real`: (Default: 2) ``p``-norm to be computed.

# Returns
`::AbstractFloat`: The relative norm between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

relativenorm(x, y)
```

**See also:** [`psnr`](@ref), [`snr`](@ref), [`ssim`](@ref)
"""
function relativenorm(x::AbstractArray{T}, 
                      x₀::AbstractArray{T}, 
                      p::Real = 2) where T<:Number
    @assert length(x) == length(x₀)             # ensure same lengths
    return norm(x-x₀,p)/norm(x₀,p)
end

# PSNR between 2 vectors
@doc raw"""
    psnr(x, x₀)

Returns the peak signal to noise ratio (PSNR) between original signal x₀ and noisy signal x.

PSNR definition: ``10 \log_{10} \frac{\max{(x_0)}^2}{MSE(x, x_0)}``

# Arguments
- `x::AbstractVector{T} where T<:Number`: Signal with noise.
- `x₀::AbstractVector{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The PSNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

psnr(x, y)
```

**See also:** [`relativenorm`](@ref), [`snr`](@ref), [`ssim`](@ref)
"""
function psnr(x::AbstractVector{T}, x₀::AbstractVector{T}) where T<:Number
    @assert length(x) == length(x₀)              # ensure same lengths
    sse = zero(T)
    for i in eachindex(x)
        @inbounds sse += (x[i] - x₀[i])^2
    end
    mse = sse/length(x)
    return 20 * log(10, maximum(x₀)) - 10 * log(10, mse)
end

# SNR between 2 vectors
@doc raw"""
    snr(x, x₀)

Returns the signal to noise ratio (SNR) between original signal x₀ and noisy signal x.

SNR definition: ``20 \log_{10} \frac{||x_0||_2}{||x-x_0||_2}``

# Arguments
- `x::AbstractVector{T} where T<:Number`: Signal with noise.
- `x₀::AbstractVector{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The SNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

snr(x, y)
```

**See also:** [`relativenorm`](@ref), [`psnr`](@ref), [`ssim`](@ref)
"""
function snr(x::AbstractVector{T}, x₀::AbstractVector{T}) where T<:Number
    @assert length(x) == length(x₀)             # ensure same lengths
    return 20 * log(10, norm(x₀,2)/norm(x-x₀,2))
end

# SSIM between 2 arrays
"""
    ssim(x, x₀)

Wrapper for `assess_ssim` function from ImageQualityIndex.jl.

Returns the Structural Similarity Index Measure (SSIM) between the original signal/image x₀
and noisy signal/image x.

# Arguments
- `x::AbstractArray{T} where T<:Number`: Signal with noise.
- `x₀::AbstractArray{T} where T<:Number`: Reference signal.

# Returns
`::AbstractFloat`: The SNR between x and x₀.

# Examples
```@repl
using WaveletsExt

x = randn(8)
y = randn(8)

ssim(x, y)
```

**See also:** [`relativenorm`](@ref), [`psnr`](@ref), [`snr`](@ref)
"""
function ssim(x::AbstractArray{T}, x₀::AbstractArray{T}) where T<:Number
    return assess_ssim(x, x₀)
end
