module SIWPD
export  
    siwpd,
    makesiwpdtree

using 
    Wavelets

using 
    ..Utils,
    ..DWT

"""
    siwpd(x, wt[, L=maxtransformlevels(x), d=L])

Computes the Shift-Invariant Wavelet Packet Decomposition originally developed
by Cohen, Raz & Malah on the vector `x` using the discrete wavelet filter `wt`
for `L` levels with depth `d`.
"""
function siwpd(x::AbstractVector{T}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x), d::Integer=L) where T<:Number
    
    g, h = WT.makereverseqmfpair(wt, true)       
    si = Vector{T}(undef, length(wt)-1)                 # tmp filter vector              
    return siwpd(x, wt, h, g, si, L, d)
end

function siwpd(x::AbstractVector{Tx}, wt::OrthoFilter, h::Vector{T}, 
        g::Vector{T}, si::Vector{T}, L::Integer, d::Integer=L) where 
        {Tx<:Number, T<:Number}

    n = length(x)
    @assert isdyadic(n)
    @assert 0 <= L <= maxtransformlevels(n)
    @assert 1 <= d <= L

    # decomposed array y 
    y = Array{T,2}(undef, (n,2^(L+1)-1))
    y[:,1] = x

    for node in axes(y,2)
        level = floor(Int, log2(node)) # current node level (starts from 0)
        nodelen = nodelength(n, level)
        # decomposition depth of current node
        dₗ = (0 <= level <= L-d) ? d : L-level  
        decompose!(y, wt, h, g, si, node, nodelen, 
            level, dₗ, 0, zeroshiftdecomposition=false)
    end

    return y
end

function decompose!(y::AbstractArray{Ty,2}, wt::OrthoFilter, 
        h::Vector{T}, g::Vector{T}, si::Vector{T}, 
        node::Integer, nodelen::Integer, level::Integer, d::Integer, 
        shift::Integer; zeroshiftdecomposition::Bool=true) where 
        {Ty<:Number, T<:Number}

    # base case
    @assert d > 0 || return nothing
    @assert 0 <= level <= maxtransformlevels(size(y,1))           
    @assert d <= maxtransformlevels(size(y,2)+1) - level        
    @assert 0 <= shift < size(y,1)                                                       
    filtlen = length(wt)
    
    pstart = shift*nodelen + 1                                  # start index
    pend = (shift+1)*nodelen                                    # end index
    parent = y[pstart:pend, node]

    cnodelen = nodelen ÷ 2
    cstart = shift*cnodelen + 1
    lchild = @view y[:, getchildindex(node,:left)]
    rchild = @view y[:, getchildindex(node,:right)]

    # non-shifted subtree
    Wavelets.Transforms.filtdown!(      # scaling coefficients
        g, si, lchild, cstart, cnodelen, parent, 1, 0, false
    )   
    Wavelets.Transforms.filtdown!(      # detail coefficients
        h, si, rchild, cstart, cnodelen, parent, 1, -filtlen+1, true
    )   
    if zeroshiftdecomposition   # pursue the same thing with d-1 if shift > 0
        decompose!(                     # decompose left child
            y, wt, h, g, si, getchildindex(node,:left), cnodelen, level+1, d-1, shift
        )   
        decompose!(                     # decompose right child
            y, wt, h, g, si, getchildindex(node,:right), cnodelen, level+1, d-1, shift
        )   
    end

    # shifted subtree
    parent = circshift(parent,1)
    cstart = (shift + 1<<level)*cnodelen + 1
    Wavelets.Transforms.filtdown!(      # scaling coefficients
        g, si, lchild, cstart, cnodelen, parent, 1, 0, false
    )
    Wavelets.Transforms.filtdown!(      # detail coefficients
        h, si, rchild, cstart, cnodelen, parent, 1, -filtlen+1, true
    )
    # pursue the same thing with d-1
    decompose!(                         # decompose shifted left child
        y, wt, h, g, si, getchildindex(node,:left), cnodelen, level+1, d-1, shift+1<<level
    )       
    decompose!(                         # decomposed shifted right child
        y, wt, h, g, si, getchildindex(node,:right), cnodelen, level+1, d-1, shift+1<<level
    )     

    return nothing
end

"""
    makesiwpdtree(n, L, d)

Returns the multi-level, multi-depth binary tree corresponding to the Shift-
Invariant Wavelet Packet Decomposition. 
"""
function makesiwpdtree(n::Integer, L::Integer, d::Integer)
    @assert 0 ≤ L ≤ maxtransformlevels(n)
    @assert 1 ≤ d ≤ L

    tree = Vector{BitVector}(undef, 2^(L+1)-1)
    for i in eachindex(tree)
        level = floor(Int, log2(i))   
        len = nodelength(n, level)   
        nshift = n ÷ len        # number of possible shifts for subspace Ω(i,j)
        node = falses(nshift)                   
        k = ceil(Int, nshift / 1<<d)  # "skips" between shifts
        node[1:k:end] .= true         # mark nodes present in tree
        tree[i] = node
    end
    return tree
end

end # end module