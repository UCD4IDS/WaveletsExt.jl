module WPD
export 
    wpd,
    wpd!,
    dwtall,
    wptall,
    wpdall

using 
    Wavelets

using 
    ..Utils

"""
    wpd(x, wt[, L=maxtransformlevels(x)])

Returns the wavelet packet decomposition (WPD) for L levels for input signal(s) 
x.

**See also:** [`wpd!`](@ref)
"""
function wpd end

"""
    wpd!(y, x, wt[, L=maxtransformlevels(x)])

Same as `wpd` but without array allocation.

**See also:** [`wpd`](@ref)
"""
function wpd! end

"""
    wpt(x, wt[, L])

    wpt(x, wt, tree)

Returns the wavelet packet transform (WPT) by `L` levels or by given quadratic `tree`.

**See also:** [`wpt!`](@ref)
"""
function Wavelets.wpt end

"""
    wpt!(y, x, wt[, L])

    wpt!(y, x, wt, tree)

Same as `wpt` but without array allocation.

**See also:** [`wpt`](@ref)
"""
function Wavelets.wpt! end

# ========== Wavelet Packet Decomposition ==========
# 1D Wavelet Packet Decomposition without allocated output array
function wpd(x::AbstractVector{T}, 
             wt::OrthoFilter, 
             L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    @assert isdyadic(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)

    # Allocate variable for result
    n = length(x)
    xw = Array{T,2}(undef, (n,L+1))
    # Wavelet packet decomposition
    wpd!(xw, x, wt, L)
    return xw
end

# 2D Wavelet Packet Decomposition without allocated output array
function wpd(x::AbstractArray{T,2},
             wt::OrthoFilter,
             L::Integer = maxtransformlevels(x);
             standard::Bool = true) where T<:Number
    # Sanity check
    @assert 0 ≤ L ≤ maxtransformlevels(x)

    # Allocate variable for result
    sz = size(x)
    xw = Array{T,3}(undef, (sz...,L+1))
    # Wavelet packet decomposition
    wpd!(xw, w, wt, L, standard=standard)
    return xw
end

# 1D Wavelet Packet Decomposition with allocated output array
function wpd!(y::AbstractArray{T,2}, 
              x::AbstractArray{T,1},
              wt::DiscreteWavelet, 
              L::Integer=maxtransformlevels(x)) where T<:Number
    # Sanity check
    n = length(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)
    @assert size(y) == (n,L+1)

    # Construct low pass and high pass filters
    gqf, hqf = WT.makereverseqmfpair(wt, true)
    # First column of y is level 0, ie. original signal
    y[:,1] = x
    # Allocate placeholder variable
    si = similar(gqf, eltype(gqf), length(wt)-1)
    # Compute L levels of decomposition
    for i in 0:(L-1)
        # Parent node length
        nₚ = nodelength(n, i) 
        for j in 0:((1<<i)-1)
            # Extract parent node
            colₚ = i + 1
            rng = (j * nₚ + 1):((j + 1) * nₚ)
            @inbounds nodeₚ = @view y[rng, colₚ]
            # Extract left and right child nodes
            colₘ = colₚ + 1
            @inbounds nodeₘ = @view y[rng, colₘ]
            # Perform 1 level of wavelet decomposition
            Transforms.unsafe_dwt1level!(nodeₘ, nodeₚ, wt, true, hqf, gqf, si)
        end
    end
    return nothing
end

# 2D Wavelet Packet Decomposition with allocated output array
function wpd!(y::AbstractArray{T,3},
              x::AbstractArray{T,2},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(x);
              standard::Bool = true) where T<:Number
    # Sanity check
    m, n = size(x)
    @assert 0 ≤ L ≤ maxtransformlevels(x)
    @assert size(y) == (m, n,L+1)

    # ----- Allocations and setup to match Wavelets.jl's function requirements -----
    fw = true                                               # forward transform
    si = Vector{T}(undef, length(wt)-1)                     # temp filter vector
    scfilter, dcfilter = WT.makereverseqmfpair(wt, fw, T)   # low & high pass filters
    # First slice of y is level 0, ie. original signal
    y[:,:,1] = x
    # ----- Compute L levels of decomposition -----
    for i in 0:(L-1)
        # Parent node width and height
        mₚ = nodelength(m, i)
        nₚ = nodelength(n, i)
        # Iterate over each nodes at current level
        lrange = 0:((1<<i)-1)
        for j in lrange
            for k in lrange
                # Extract parent node
                sliceₚ = i+1
                rng_row = (j*mₚ+1):((j+1)*mₚ)
                rng_col = (k*nₚ+1):((k+1)*nₚ)
                @inbounds nodeₚ = @view y[rng_row, rng_col, sliceₚ]
                # Extract children nodes of current parent
                sliceᵣ = sliceₚ+1
                @inbounds nodeᵣ = @view y[rng_row, rng_col, sliceᵣ]
                # Perform 1 level of wavelet decomposition
                dwt_step!(nodeᵣ, nodeₚ, wt, dcfilter, scfilter, si, standard=standard)
            end
        end
    end
    return y
end

# ========== Wavelet Packet Transform ==========
# TODO: Think of a good way to represent the best basis tree for 2D transforms
# TODO: Suggestion: [root, LL, LH, HL, HH, LLLL, LLLH, LLHL, LLHH, LHLL, LHLH, ...]
# TODO: Add `makequadtree` function in Utils
# TODO: Add `isvalidquadtree` function in Utils
# TODO: Figure out how to get tree level
# 2D Wavelet Packet Transform without allocated output array
function Wavelets.Transforms.wpt(x::AbstractArray{T,2}, 
                                 wt::OrthoFilter, 
                                 L::Integer = maxtransformlevels(x),
                                 standard::Bool = true) where T<:Number
    return wpt(x, wt, makequadtree(x, L, :full), standard=standard)
end

function Wavelets.Transforms.wpt(x::AbstractArray{T,2},
                                 wt::OrthoFilter,
                                 tree::BitVector;
                                 standard::Bool = true) where T<:Number
    y = Array{T}(undef, size(x))
    return wpt!(y, x, wt, tree, standard=standard)
end

# 2D Wavelet Packet Transform with allocated output array
function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  L::Integer = maxtransformlevels(x),
                                  standard::Bool = true) where T<:Number
    return wpt!(y, x, wt, makequadtree(x, L, :full), standard=standard)
end

function Wavelets.Transforms.wpt!(y::AbstractArray{T,2},
                                  x::AbstractArray{T,2},
                                  wt::OrthoFilter,
                                  tree::BitVector;
                                  standard::Bool = true) where T<:Number
    # Sanity check
    @assert isvalidquadtree(x, tree)
    @assert size(y) == size(x)

    # ----- Allocation and setup to match Wavelets.jl's function requirements -----
    m, n = size(x)
    fw = true
    si = Vector{T}(undef, length(wt)-1)                     # temp filter vector
    scfilter, dcfilter = WT.makereverseqmfpair(wt, fw, T)   # low & high pass filters
    temp = x                                                # temp array
    # ----- Compute transforms based on tree -----
    for i in eachindex(tree)
        # Decompose if node i has children
        if tree[i]
            # TODO: Figure out how to get tree level
            # Compute tree level
            level = treelevel(i)
            # Parent node width and height
            mₚ = nodelength(m, level)
            nₚ = nodelength(n, level)
            # Extract parent node

            # Extract children nodes of current parent

            # Perform 1 level of wavelet decomposition
            dwt_step!(nodeᵣ, nodeₚ, wt, dcfilter, scfilter, si, standard=standard)
        else
            continue
        end
    end
end

include("wt_one_level.jl")
include("wt_all.jl")

end # end module