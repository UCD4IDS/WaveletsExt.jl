module SIWPD
export  
    siwpd,
    siwpd_tree

using 
    Wavelets

using 
    ..Utils,
    ..WPD

# siwpd function
## TODO: figure out what to do if i set d=0
function siwpd(x::AbstractArray{T}, filter::OrthoFilter, L::Integer=maxtransformlevels(x), d::Integer=L) where T<:Number
    scfilter, dcfilter = WT.makereverseqmfpair(filter, true)       
    si = Vector{T}(undef, length(filter)-1)                 # tmp filter vector              
    return siwpd(x, filter, dcfilter, scfilter, si, L, d)
end

function siwpd(x::AbstractVector{Tx}, filter::OrthoFilter, dcfilter::Vector{T}, scfilter::Vector{T}, si::Vector{T}, 
    L::Integer=maxtransformlevels(x), d::Integer=L) where {Tx<:Number, T<:Number}

    n = length(x)
    @assert isdyadic(n)
    @assert 0 <= L <= maxtransformlevels(n)
    @assert 1 <= d <= L

    # decomposed array y 
    y = Array{T,2}(undef, (n,2^(L+1)-1))
    y[:,1] = x

    for node in axes(y,2)
        level = floor(Int, log2(node))          # level of current node (top level starts from 0)
        nodelen = nodelength(n, level)         # length of current node
        numnode = n ÷ nodelen                   # number of nodes at current level
        dₗ = (0 <= level <= L-d) ? d : L-level  # decomposition depth of current node
        decompose!(y, filter, dcfilter, scfilter, si, node, nodelen, level, dₗ, 0, zeroshiftdecomposition=false) # decompose current node
    end

    return y
end

function decompose!(y::AbstractArray{Ty,2}, filter::OrthoFilter, dcfilter::Vector{T}, scfilter::Vector{T}, si::Vector{T}, 
    node::Integer, nodelen::Integer, level::Integer, d::Integer, shift::Integer;
    zeroshiftdecomposition::Bool=true) where {Ty<:Number, T<:Number}

    # base case
    @assert d > 0 || return nothing
    @assert 0 <= level <= maxtransformlevels(size(y,1))           # ensure level is valid
    @assert d <= maxtransformlevels(size(y,2)+1) - level        # ensure depth d <= remaining levels in the tree
    @assert 0 <= shift < size(y,1)                               # ensure shift is valid
    filtlen = length(filter)
    
    pstart = shift*nodelen + 1                                  # start index
    pend = (shift+1)*nodelen                                    # end index
    parent = y[pstart:pend, node]

    cnodelen = nodelen ÷ 2
    cstart = shift*cnodelen + 1
    lchild = @view y[:, node<<1]
    rchild = @view y[:, node<<1+1]

    # non-shifted subtree
    # scaling coefficients
    Wavelets.Transforms.filtdown!(scfilter, si, lchild, cstart, cnodelen, parent, 1, 0, false)
    # detail coefficients
    Wavelets.Transforms.filtdown!(dcfilter, si, rchild, cstart, cnodelen, parent, 1, -filtlen+1, true)
    # pursue the same thing with d-1 if shift > 0
    if zeroshiftdecomposition        
        decompose!(y, filter, dcfilter, scfilter, si, node<<1, cnodelen, level+1, d-1, shift)   # decompose left child
        decompose!(y, filter, dcfilter, scfilter, si, node<<1+1, cnodelen, level+1, d-1, shift) # decompose right child
    end

    # shifted subtree
    parent = circshift(parent,1)
    cstart = (shift + 1<<level)*cnodelen + 1
    # scaling coefficients
    Wavelets.Transforms.filtdown!(scfilter, si, lchild, cstart, cnodelen, parent, 1, 0, false)
    # detail coefficients
    Wavelets.Transforms.filtdown!(dcfilter, si, rchild, cstart, cnodelen, parent, 1, -filtlen+1, true)
    # pursue the same thing with d-1
    decompose!(y, filter, dcfilter, scfilter, si, node<<1, cnodelen, level+1, d-1, shift + 1<<level)       # decompose shifted left child
    decompose!(y, filter, dcfilter, scfilter, si, node<<1+1, cnodelen, level+1, d-1, shift + 1<<level)     # decomposed shifted right child

    return nothing
end

# siwpd tree
function siwpd_tree(n::Integer, L::Integer, d::Integer)
    @assert 0 <= L <= maxtransformlevels(n)
    @assert 1 <= d <= L

    tree = Vector{BitVector}(undef, 2^(L+1)-1)
    for i in eachindex(tree)
        level = floor(Int, log2(i))             # level of node i
        len = nodelength(n, level)             # length of node
        nshift = n ÷ len                        # number of available shifts corresponding to the subspace Ω(i,j)
        node = falses(nshift)                   
        k = ceil(Int, nshift / 1<<d)            # "skips" between shifts to denote the shifts that were computed in siwpd
        node[1:k:end] .= true                   # mark computed shifts in siwpd as `true`
        tree[i] = node
    end
    return tree
end

end # end module