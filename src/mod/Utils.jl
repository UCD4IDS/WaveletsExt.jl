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
    generatesignals

using
    Wavelets,
    LinearAlgebra,
    Random

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
        if i<<1 > length(result)
            break
        elseif tree[i] == 0
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
    coarsestscalingrange(x, tree[, stationary=false])

    coarsestscalingrange(n, tree[, stationary=false])

Given a binary tree, returns the index range of the coarsest scaling 
coefficients.
"""
function coarsestscalingrange(x::AbstractArray{T}, tree::BitVector, 
        stationary::Bool=false) where T<:Number

    return coarsestscalingrange(size(x,1), tree, stationary)
end

function coarsestscalingrange(n::Integer, tree::BitVector, 
        stationary::Bool=false)

    if !stationary          # regular wt
        i = 1
        j = 0
        while i<length(tree) && tree[i]       # has children
            i = left(i)
            j += 1
        end
        rng = 1:(n>>j) 
    else                    # stationary wt
        i = 1
        while i<length(tree) && tree[i]       # has children
            i = left(i)
        end
        rng = (1:n, i)
    end
    return rng
end

"""
    finestdetailrange(x, tree[, stationary=false])
    
    finestdetailrange(n, tree[, stationary=false])

Given a binary tree, returns the index range of the coarsest scaling 
coefficients.
"""
function finestdetailrange(x::AbstractArray{T}, tree::BitVector,
        stationary::Bool=false) where T<:Number

    return finestdetailrange(size(x,1), tree, stationary)
end

function finestdetailrange(n::Integer, tree::BitVector, stationary::Bool=false)
    if !stationary      # regular wt
        i = 1
        j = 0
        while i<length(tree) && tree[i]
            i = right(i)
            j += 1
        end
        n₀ = nodelength(n, j)
        rng = (n-n₀+1):n
    else                # stationary wt
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

end # end module