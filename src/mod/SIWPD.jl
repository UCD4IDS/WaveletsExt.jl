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
               L::S = maxtransformlevels(x), d::S = L) where {T<:AbstractFloat, S<:Integer}
    # Sanity check
    n = length(x)
    @assert 0 ≤ L ≤ maxtransformlevels(n)
    @assert 1 ≤ d ≤ L

    # Setup
    g, h = WT.makereverseqmfpair(wt, true)
    # y = Matrix{T}(undef, (n, gettreelength(1<<(L+1))))
    y = zeros(T, (n, gettreelength(1<<(L+1))))
    @inbounds y[:,1] = x

    # Decomposition
    for i in axes(y,2)
        lvl = getdepth(i, :binary)
        len = nodelength(n, lvl)
        dₗ = (0 ≤ lvl ≤ L-d) ? d : L-lvl
        siwpd_subtree!(y, h, g, i, dₗ, 0, L=lvl, len=len)
    end
    return y
end

#=======================================================================================
Recursive calls for decomposition while d>0.
If shift is 0, ie. ss=0, the main root will not be further decomposed even if d>0.
=======================================================================================#
function siwpd_subtree!(y::AbstractMatrix{T₁}, h::Vector{T₂}, g::Vector{T₂}, 
                        i::S, d::S, ss::S; 
                        L::S = getdepth(i, :binary),
                        len::S = nodelength(size(y,1), L)) where 
                       {T₁<:AbstractFloat, T₂<:AbstractFloat, S<:Integer}
    # Sanity check
    n, k = size(y)
    @assert 0 ≤ L ≤ maxtransformlevels(n)
    @assert 0 ≤ ss < n
    @assert 0 ≤ d ≤ getdepth(k,:binary)-L
    @assert iseven(len) || len == 1

    # --- Base case ---
    d > 0 || return y

    # --- One level of decomposition ---
    v_start = ss*len+1                                      # Start index for parent node
    v_end = (ss+1)*len                                      # End index for parent node
    @inbounds v = @view y[v_start:v_end, i]                 # Parent node
    w_start = ss*(len÷2)+1                                  # Start index for child node
    w_end = w_start+(len÷2)-1                               # End index for child node
    @inbounds w₁ = @view y[w_start:w_end, getchildindex(i, :left)]    # Left child node
    @inbounds w₂ = @view y[w_start:w_end, getchildindex(i, :right)]   # Right child node
    sidwt_step!(w₁, w₂, v, h, g, false)                          # 1 step of non-shifted decomposition
    w_start = (ss+1<<L)*(len÷2)+1                           # Start index for shifted child node
    w_end = w_start+(len÷2)-1                               # End index for shifted child node
    @inbounds w₁ = @view y[w_start:w_end, getchildindex(i, :left)]    # Left child node
    @inbounds w₂ = @view y[w_start:w_end, getchildindex(i, :right)]   # Right child node
    sidwt_step!(w₁, w₂, v, h, g, true)                           # 1 step of shifted decomposition
    
    # --- Recursive decomposition while d>0 ---
    if ss > 0       # Decomposition of current shift
        siwpd_subtree!(y, h, g, getchildindex(i, :left), d-1, ss, L=L+1, len=len÷2)
        siwpd_subtree!(y, h, g, getchildindex(i, :right), d-1, ss, L=L+1, len=len÷2)
    end
    # Decomposition of shifted version
    siwpd_subtree!(y, h, g, getchildindex(i, :left), d-1, ss+1<<L, L=L+1, len=len÷2)
    siwpd_subtree!(y, h, g, getchildindex(i, :right), d-1, ss+1<<L, L=L+1, len=len÷2)

    return y
end

# Single step of decomposition
function sidwt_step!(w₁::AbstractVector{T}, w₂::AbstractVector{T},
                v::AbstractVector{T},
                h::Vector{S}, g::Vector{S},
                s::Bool) where {T<:AbstractFloat, S<:AbstractFloat}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)÷2
    @assert length(h) == length(g)

    # Setup
    n = length(v)           # Parent length
    n₁ = length(w₁)         # Child length
    filtlen = length(h)     # Filter length

    # One step of discrete transform
    for i in 1:n₁
        k₁ = mod1(2*i-1-s, n)   # Start index for low pass filtering
        k₂ = 2*i-s              # Start index for high pass filtering
        @inbounds w₁[i] = g[end] * v[k₁]
        @inbounds w₂[i] = h[1] * v[k₂]
        for j in 2:filtlen
            k₁ = k₁+1 |> k₁ -> k₁>n ? mod1(k₁,n) : k₁
            k₂ = k₂-1 |> k₂ -> k₂≤0 ? mod1(k₂,n) : k₂
            @inbounds w₁[i] += g[end-j+1] * v[k₁]
            @inbounds w₂[i] += h[j] * v[k₂]
        end
    end
    return w₁, w₂
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