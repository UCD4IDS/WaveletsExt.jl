module WPD
export 
    wpd,
    wpd!

using 
    Wavelets

using 
    ..Utils

"""
    wpd(x, wt[, L=maxtransformlevels(x)])

    wpd(x, wt, hqf, gqf[, L=maxtransformlevels(x)])

Returns the wavelet packet decomposition WPD) for L levels for input signal(s) 
x.
"""
function wpd(x::AbstractVector{<:Number}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x))

    gqf, hqf = WT.makereverseqmfpair(wt, true)
    
    return wpd(x, wt, hqf, gqf, L)
end

function wpd(x::AbstractVector{T}, wt::DiscreteWavelet, hqf::Array{S,1}, 
        gqf::Array{S,1}, L::Integer=maxtransformlevels(x)) where 
        {T<:Number, S<:Number}

    n = length(x)
    @assert isdyadic(n)
    @assert 0 <= L <= maxtransformlevels(n)

    result = Array{T, 2}(undef, (n, L+1))
    wpd!(result, x, wt, hqf, gqf, L)
    return result
end


"""
    wpd!(y, x, wt[, L=maxtransformlevels(x)])

    wpd!(y, x, wt, hqf, gqf[, L=maxtransformlevels(x)])

Same as `wpd` but without array allocation.
"""
function wpd!(y::AbstractArray{T,2}, x::AbstractArray{T,1},
        wt::DiscreteWavelet, L::Integer=maxtransformlevels(x)) where
        T<:Number

    gqf, hqf = WT.makereverseqmfpair(wt, true)
    wpd!(y, x, wt, hqf, gqf, L)
    return nothing
end

function wpd!(y::AbstractArray{T,2}, x::AbstractArray{T,1}, 
        wt::DiscreteWavelet, hqf::Array{S,1}, gqf::Array{S,1}, 
        L::Integer=maxtransformlevels(x)) where {T<:Number, S<:Number}

    n = length(x)
    @assert 0 <= L <= maxtransformlevels(n)
    @assert size(y) == (n, L+1)

    y[:,1] = x
    si = Vector{S}(undef, length(wt)-1)
    for i in 0:(L-1)
        # parent node length
        nₚ = nodelength(n, i) 
        for j in 0:((1<<i)-1)
            # extract parent node
            colₚ = i + 1
            rng = (j * nₚ + 1):((j + 1) * nₚ)
            @inbounds nodeₚ = @view y[rng, colₚ]

            # extract left and right child nodes
            colₘ = colₚ + 1
            @inbounds nodeₘ = @view y[rng, colₘ]

            # perform 1 level of wavelet decomposition
            Transforms.unsafe_dwt1level!(nodeₘ, nodeₚ, wt, true, hqf, gqf, si)
        end
    end
    return nothing
end

end # end module