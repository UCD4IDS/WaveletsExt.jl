module Utils
export 
    left,
    right,
    nodelength,
    getleaf,
    coarsestscalingrange,
    finestdetailrange,
    relativenorm,
    psnr,
    snr,
    ssim,
    generatesignals

using
    Wavelets,
    LinearAlgebra,
    Random, 
    ImageQualityIndexes

"""
    left(i)

Given the node index `i`, returns the index of its left node.
"""
left(i::Integer) = i<<1

"""
    right(i)

Given the node index `i`, returns the index of its right node.
"""
right(i::Integer) = i<<1 + 1

"""
    nodelength(N, L)

Returns the node length at level L of a signal of length N. Level L == 0 
corresponds to the original input signal.
"""
function nodelength(N::Integer, L::Integer)
    return (N >> L)
end

"""
    getleaf(tree)

Returns the leaf nodes of a tree.
"""
function getleaf(tree::BitVector)
    @assert isdyadic(length(tree) + 1)

    result = falses(2*length(tree) + 1)
    result[1] = 1
    for i in 1:length(tree)
        if tree[i] == 0
            continue
        else
            result[i] = 0
            result[i<<1] = 1
            result[i<<1 + 1] = 1
        end
    end
    
    return result
end

"""
    coarsestscalingrange(x, tree[, redundant=false])

    coarsestscalingrange(n, tree[, redundant=false])

Given a binary tree, returns the index range of the coarsest scaling 
coefficients.
"""
function coarsestscalingrange(x::AbstractArray{T}, tree::BitVector, 
        redundant::Bool=false) where T<:Number

    return coarsestscalingrange(size(x,1), tree, redundant)
end

function coarsestscalingrange(n::Integer, tree::BitVector, 
        redundant::Bool=false)

    if !redundant          # regular wt
        i = 1
        j = 0
        while i<length(tree) && tree[i]       # has children
            i = left(i)
            j += 1
        end
        rng = 1:(n>>j) 
    else                   # redundant wt
        i = 1
        while i<length(tree) && tree[i]       # has children
            i = left(i)
        end
        rng = (1:n, i)
    end
    return rng
end

"""
    finestdetailrange(x, tree[, redundant=false])
    
    finestdetailrange(n, tree[, redundant=false])

Given a binary tree, returns the index range of the coarsest scaling 
coefficients.
"""
function finestdetailrange(x::AbstractArray{T}, tree::BitVector,
        redundant::Bool=false) where T<:Number

    return finestdetailrange(size(x,1), tree, redundant)
end

function finestdetailrange(n::Integer, tree::BitVector, redundant::Bool=false)
    if !redundant      # regular wt
        i = 1
        j = 0
        while i<length(tree) && tree[i]
            i = right(i)
            j += 1
        end
        n₀ = nodelength(n, j)
        rng = (n-n₀+1):n
    else               # redundant wt
        i = 1
        while i<length(tree) && tree[i]
            i = right(i)
        end
        rng = (1:n, i)
    end
    return rng
end

"""
    relativenorm(x, x₀[, p=2]) where T<:Number

Returns the relative norm of base p between original signal x₀ and noisy signal
x.
"""
function relativenorm(x::AbstractVector{T}, x₀::AbstractVector{T}, 
        p::Real=2) where T<:Number

    @assert length(x) == length(x₀)             # ensure same lengths
    return norm(x-x₀,p)/norm(x₀,p)
end

"""
    psnr(x, x₀)

Returns the peak signal to noise ratio (PSNR) between original signal x₀ and
noisy signal x.
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

"""
    snr(x, x₀)

Returns the signal to noise ratio (SNR) between original signal x₀ and noisy 
signal x.
"""
function snr(x::AbstractVector{T}, x₀::AbstractVector{T}) where T<:Number
    @assert length(x) == length(x₀)             # ensure same lengths
    return 20 * log(10, norm(x₀,2)/norm(x-x₀,2))
end

"""
    ssim(x, x₀)

Wrapper for `assess_ssim` function from ImageQualityIndex.jl.

Returns the Structural Similarity Index Measure (SSIM) between the original 
signal/image x₀ and noisy signal/image x. 
"""
function ssim(x::AbstractArray{T}, x₀::AbstractArray{T}) where T<:Number
    return assess_ssim(x, x₀)
end

"""
    generatesignals(x, N, k[, noise=false, t=1])

Given a signal x, returns N shifted versions of the signal, each with shifts
of multiples of k. 

Setting `noise = true` allows randomly generated Gaussian noises of μ = 0, 
σ² = t to be added to the circularly shifted signals.
"""
function generatesignals(x::AbstractVector{T}, N::Integer, k::Integer, 
        noise::Bool=false, t::Real=1) where T<:Number

    n = length(x)
    X = Array{T, 2}(undef, (n, N))
    for i in 1:N
        X[:,i] = circshift(x, k*(i-1)) 
    end
    X = noise ? X + t*randn(n,N) : X
    return X
end

function generatefunction(fn::Symbol, L::Integer)
    @assert L >= 1

    t = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
    h = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2]
    
    n = 1<<L
    if fn == :blocks
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j]*(1 .+ sign.(tt .- t[j]))/2)
        end
    elseif fn == :bumps
        h = abs.(h)
        w = 0.01*[0.5, 0.5, 0.6, 1, 1, 3, 1, 1, 0.5, 0.8, 0.5]
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j] ./ (1 .+ ((tt .- t[j]) / w[j]).^4))
        end
    elseif fn == :heavysine
        x = collect(range(0, 1, length=n))
        x = 4*sin.(4*pi*x) - sign.(x .- 0.3) - sign.(0.72 .- x)
    elseif fn == :doppler
        x = collect(range(0, 1, length=n))
        ϵ = 0.05
        x = sqrt.(x.*(1 .- x)) .* sin.(2*pi*(1+ϵ) ./ (x.+ϵ))
    elseif fn == :quadchirp
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
    elseif fn == :mishmash
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
        x = x + sin.(π * (n * 0.6902) * tt)
        x = x + sin.(π * tt .* (n * 0.125 * tt))
    else
        throw(ArgumentError("Unrecognised `fn`. Type `?generatefunction` to learn more."))
    end
    return x
end

h₁(i::Int) = max(6 - abs(i-7), 0)
h₂(i::Int) = h₁(i - 8)
h₃(i::Int) = h₁(i - 4)

"""
    generatetriangular(c1::Int, c2::Int, c3::Int, L::Int=32)

Generates a set of triangluar test functions with 3 classes.
"""
function generatetriangular(c1::Int, c2::Int, c3::Int; L::Int=32, shuffle::Bool=false)
    @assert c1 >= 0
    @assert c2 >= 0
    @assert c3 >= 0
  
    u = rand(Uniform(0,1),1)[1]
    ϵ = rand(Normal(0,1),(L,c1+c2+c3))
  
    y = vcat(ones(c1), ones(c2) .+ 1, ones(c3) .+ 2)
  
    H₁ = Array{Float64,2}(undef,L,c1)
    H₂ = Array{Float64,2}(undef,L,c2)
    H₃ = Array{Float64,2}(undef,L,c3)
    for i in 1:L
      H₁[i,:] .= u * h₁(i) + (1 - u) * h₂(i)
      H₂[i,:] .= u * h₁(i) + (1 - u) * h₃(i)
      H₃[i,:] .= u * h₂(i) + (1 - u) * h₃(i)
    end
  
    H = hcat(H₁, H₂, H₃) + ϵ
  
    if shuffle
      idx = [1:(c1+c2+c3)...]
      shuffle!(idx)
      return H[:,idx], y[idx]
    end
  
    return H, y
end

end # end module