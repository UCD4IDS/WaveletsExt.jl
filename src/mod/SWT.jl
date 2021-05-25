module SWT

export 
    # stationary wavelet transform
    sdwt,
    swpd,
    swpt,
    # inverse stationary wavelet transform
    isdwt,
    iswpt

using Wavelets

"""
    sdwt_step(v, j, h, g)

Perform one level of the stationary discrete wavelet transform (SDWT) on the 
vector `v`, which is the j-th level scaling coefficients (Note the 0th level
scaling coefficients is the raw signal). The vectors `h` and `g` are the detail
and scaling filters.

Returns a tuple `(v, w)` of the scaling and detail coefficients at level `j+1`.

**See also:** [`sdwt_step!`](@ref)
"""
function sdwt_step end

"""
    sdwt_step!(v1, w1, v, j, h, g)

Same as `sdwt_step` but without array allocation.

**See also:** [`sdwt_step`](@ref)
"""
function sdwt_step! end

"""
    sdwt(x, wt[, L=maxtransformlevels(x)])

Perform a stationary discrete wavelet transform (SDWT) of the array `x`. The
wavelet type `wt` determines the transform type and the wavelet class, see 
`wavelet`. 

The number of transform levels `L` can be 1 ≤ L ≤ `maxtransformlevels(x)`.
Default value is set to `maxtransformlevels(x)`.

Returns the `n × (L+1)` matrix (where `n` is the length of `x`) with the detail
coefficients for level j in column (L-j+2). The scaling coefficients are in the
1st column.

# Examples
```julia
xw = sdwt(x, wt, 5)
```

**See also:** [`swpd`](@ref), [`swpt`](@ref), [`isdwt`](@ref)
"""
function sdwt end

"""
    swpd(x, wt[, L=maxtransformlevels(x)])

Perform a stationary wavelet packet decomposition (SPWD) of the array `x`. The 
wavelet type `wt` determines the transform type and the wavelet class, see 
`wavelet`.

The number of transform levels `L` can be 1 ≤ L ≤ `maxtransformlevels(x)`.
Default value is set to `maxtransformlevels(x)`.

Returns the `n × (2⁽ᴸ⁺¹⁾-1)` matrix (where `n` is the length of `x`) with each 
column representing the nodes in the binary tree.

# Examples
```julia
xw = swpd(x, wt, 5)
```


**See also:** [`swpt`](@ref), [`sdwt`](@ref), [`iswpt`](@ref)
"""
function swpd end

"""
    swpt(x, wt[, L=maxtransformlevels(x)])

    swpt(x, wt, tree)

    swpt(x, h, g, tree)

    swpt(x, h, g, tree, i)

Performs the stationary wavelet packet transform (SWPT) of the vector `x` of 
length `N = 2ᴸ`. The wavelet type `wt` determines the transform type and the 
wavelet class, see `wavelet`.

The number of transform levels `L` can be 1 ≤ L ≤ `maxtransformlevels(x)`.
Default value is set to `maxtransformlevels(x)`.

Returns the expansion coefficients of the SWPT of the size `N × k`. Each column 
represents a leaf node from `tree`. Number of returned columns can vary between 
1 ≤ k ≤ N depending on the input `tree`.

*Note:* If one wants to compute the stationary wavelet packet transform on a 
signal, yet hopes to reconstruct the original signal later on, please use the 
`swpd` function instead.

**See also:** [`sdwt`](@ref), [`swpd`](@ref)
"""
function swpt end

"""
    isdwt_step(v1, w1, j, s0, s1, g, h)

Perform one level of the inverse stationary discrete wavelet transform (ISDWT) 
on the vector `v1` and `w1`, which is the j-th level scaling coefficients (Note 
the 0th level scaling coefficients is the raw signal). The vectors `h` and `g` 
are the detail and scaling filters.

Returns a vector `v0` of the scaling and detail coefficients at level `j-1`.

**See also:** [`isdwt_step!`](@ref)
"""
function isdwt_step end

"""
    isdwt_step!(v0, v1, w1, j, s0, s1, g, h)

Same as `isdwt_step` but without array allocation.

**See also:** [`isdwt_step`](@ref)
"""
function isdwt_step! end

"""
    isdwt(xw, wt[, ε])

Performs the inverse stationary discrete wavelet transform (ISDWT) on the `sdwt`
transform coefficients with respect to the Boolean Vector `ε` which represents 
the shifts to be used. If `ε` is not provided, the average-basis ISDWT will be 
computed instead.

# Examples
```julia
# decompose signal
xw = sdwt(x, wt, 5)

# ε-based reconstruction
y = isdwt(xw, wt, BitVector([0,1,0,0,0]))

# average-based reconstruction
y = isdwt(xw, wt)
```

**See also:** [`iswpt`](@ref), [`sdwt`](@ref)
"""
function isdwt end

"""
    iswpt(xw, wt, ε[, L=maxtransformlevels(size(xw,1))])

    iswpt(xw, wt, ε, tree)

    iswpt(xw, wt[, L=maxtransformlevels(size(xw,1))])

    iswpt(xw, wt, tree)

Performs the inverse stationary wavelet packet transform (ISWPT) on the `swpd`
transform coefficients with respect to a given Boolean Vector that represents a
binary `tree` and the BitVector `ε` which represents the shifts to be used. If
`ε` is not provided, the average-basis ISWPT will be computed instead.

# Examples
```julia
# decompose signal
xw = swpd(x, wt, 5)

# best basis tree
bt = bestbasistree(xw, BB(redundant=true))

# ε-based reconstruction
y = iswpt(xw, wt, BitVector([0,0,0,1,0]), 5)
y = iswpt(xw, wt, BitVector([0,0,0,1,0]), bt)

# average-based reconstruction
y = iswpt(xw, wt, 5)
y = iswpt(xw, wt, bt)
```

**See also:** [`isdwt`](@ref), [`swpd`](@ref)
"""
function iswpt end

# decomposition step of SDWT
function sdwt_step(v::AbstractVector{T}, j::Integer, h::Array{S,1},
        g::Array{S,1}) where {T<:Number, S<:Number}
        
    N = length(v)
    v1 = zeros(T, N)
    w1 = zeros(T, N)

    sdwt_step!(v1, w1, v, j, h, g)
    return v1, w1
end

# decomposition step of SDWT without allocation
function sdwt_step!(v1::AbstractVector{T}, w1::AbstractVector{T},
        v::AbstractVector{T}, j::Integer, h::Array{S,1},
        g::Array{S,1}) where {T<:Number, S<:Number}

    N = length(v)
    L = length(h)

    @inbounds begin
        for t in 1:N
            k = t
            w1[t] = h[1] * v[k]
            v1[t] = g[1] * v[k]
            for n in 2:L
                k += 1 << j
                if k > N
                    k = mod1(k, N)
                end
                w1[t] += h[n] * v[k]
                v1[t] += g[n] * v[k]
            end
        end
    end
    return nothing
end

# stationary discrete wavelet transform (SDWT)
function sdwt(x::AbstractVector{T}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x)) where T<:Number
    
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    g, h = WT.makeqmfpair(wt)
    N = length(x)
    W = zeros(T, (N, L))
    V = deepcopy(x)

    @inbounds begin
        for j in 0:(L-1)
            V[:], W[:, L - j] = sdwt_step(V, j, h, g)
        end    
    end
    return [V W]
end

# stationary wavelet packet decomposition (SWPD)
function swpd(x::AbstractVector{T}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x)) where T<:Number

    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    g, h = WT.makeqmfpair(wt)
    N = length(x)
    n₀ = 1 << (L+1) - 1
    W = zeros(T, (N, n₀))
    W[:, 1] = x

    @inbounds begin
        for i in 1:n₀
            if (i << 1) > n₀
                break
            end
            j = floor(Int, log2(i))
            c₁, c₂ = sdwt_step(W[:, i], j, h, g)
            W[:,i<<1], W[:,i<<1+1] = c₁, c₂
        end
    end
    return W
end

# stationary wavelet packet transform (SWPT)
function swpt(x::AbstractVector{T}, wt::DiscreteWavelet,                        
        L::Integer=maxtransformlevels(x)) where T<:Number

    return swpt(x, wt, maketree(length(x), L, :full))
end

function swpt(x::AbstractVector{T}, filter::OrthoFilter,
        tree::BitVector) where T<:Number

    g, h = WT.makeqmfpair(filter)
    return swpt(x, h, g, tree, 1)
end

function swpt(x::AbstractVector{T}, h::Array{S,1}, g::Array{S,1},
        tree::BitVector) where {T<:Number, S<:Number}

    return swpt(x, h, g, tree, 1)
end

function swpt(x::AbstractVector{T}, h::Array{S,1}, g::Array{S,1},
        tree::BitVector, i::Integer) where {T<:Number, S<:Number}

    L = length(tree)                # tree length
    @assert 0 <= i <= L
    i > 0 || return x

    j = floor(Integer, log2(i))     # current level
    v, w = sdwt_step(x, j, h, g)

    if i<<1 > L                     # leaf nodes: bottom level of tree
        return [v w]
    end

    if tree[i<<1]                   # left node has children
        v = swpt(v, h, g, tree, i<<1)
    end
    if tree[i<<1+1]                 # right node has children
        w = swpt(w, h, g, tree, i<<1+1)
    end
    
    return [v w]
end

# reconstruction step of inverse stationary discrete wavelet transform (ISDWT)
function isdwt_step(v1::AbstractVector{T}, w1::AbstractVector{T}, j::Integer,
        s0::Integer, s1::Integer, g::Array{S,1}, h::Array{S,1}) where 
        {T<:Number, S<:Number}

    v0 = zeros(T, length(v1))
    isdwt_step!(v0, v1, w1, j, s0, s1, g, h)
    return v0
end

# isdwt_step with no memory allocation
function isdwt_step!(v0::AbstractVector{T}, v1::AbstractVector{T},
        w1::AbstractVector{T}, j::Integer, s0::Integer, s1::Integer,
        g::Array{S,1}, h::Array{S,1}) where {T<:Number, S<:Number}

    N = length(v1)      # signal length
    L = length(h)       # filter length
    # parent start index and step size
    pstart = s0 + 1
    pstep = 1 << (j-1)
    # child start index and step size
    cstart = s1 + 1
    cstep = 1 << j
    # isdwt for each coefficient
    @inbounds begin
        for (t, m) in enumerate(pstart:pstep:N)
            i = rem(t, 2) == 1 ? 1 : 2
            # circshift needed if s1 > s0
            n = s1 == s0 ? m : mod1(m + pstep, N)
            # child start index for calculating v0[n]
            k = (ceil(Integer, t/2) - 1) * cstep + cstart
            #calculation of v0[n]
            v0[n] = h[i] * w1[k] + g[i] * v1[k]
            while i + 2 <= L
                k -= 1 << j
                if k <= 0
                    k = mod1(k, N)
                end
                i += 2
                v0[n] += h[i] * w1[k] + g[i] * v1[k]
            end
        end
    end
    return nothing
end

# ε-basis inverse stationary discrete wavelet transform (ISDWT)
function isdwt(xw::AbstractArray{T,2}, wt::OrthoFilter,
        ε::AbstractVector{Bool}) where T<:Number

    _, K = size(xw)
    L = K - 1
    @assert length(ε) == L

    g, h = WT.makeqmfpair(wt)
    shifts = collect(0:(L - 1))
    cumsum!(shifts, ε .<< shifts)    # shifts represented by each εᵢ
    j = L                            # current level
    x = xw[:, 1]
    @inbounds begin
        for i in 2:K
            s0 = j == 1 ? 0 : shifts[j-1]
            s1 = shifts[j]
            x = isdwt_step(x, xw[:, i], j, s0, s1, g, h)
            j -= 1
        end
    end
    return x
end

# average-basis inverse stationary discrete wavelet transform (ISDWT)
function isdwt(xw::AbstractArray{T,2}, wt::OrthoFilter) where T<:Number
    g, h = WT.makeqmfpair(wt)
    v2 = size(xw, 2) > 2 ? xw[:, 1:end .!= end] : xw[:, 1]
    w2 = xw[:, end]
    v1₀ = isdwt(v2, w2, 1, 0, 0, g, h)
    v1₁ = isdwt(v2, w2, 1, 0, 1, g, h)
    return (v1₀ + v1₁) / 2
end

function isdwt(v::AbstractArray{T,2}, w1::AbstractArray{T,1}, j::Integer,
        s0::Integer, s1::Integer, g::Array{S,1}, h::Array{S,1}) where
        {T<:Number, S<:Number}

    v2 = size(v, 2) > 2 ? v[:, 1:end .!= end] : v[:, 1]
    w2 = v[:, end]
    v1₀ = isdwt(v2, w2, j + 1, s1, s1, g, h)
    v1₁ = isdwt(v2, w2, j + 1, s1, s1 + 1<<j, g, h)
    v1 = (v1₀ + v1₁) / 2

    return isdwt_step(v1, w1, j, s0, s1, g, h)
end

function isdwt(v1::AbstractArray{T,1}, w1::AbstractArray{T,1}, j::Integer,
        s0::Integer, s1::Integer, g::Array{S,1}, h::Array{S,1}) where
        {T<:Number, S<:Number}

    return isdwt_step(v1, w1, j, s0, s1, g, h)
end

# ε-basis inverse stationary wavelet packet transform (ISWPT)
function iswpt(xw::AbstractArray{T,2}, wt::DiscreteWavelet, 
        ε::AbstractVector{Bool}, 
        L::Integer=maxtransformlevels(size(xw,1))) where T<:Number

    @assert length(ε) == L
    return iswpt(xw, wt, ε, maketree(size(xw,1), L, :full))
end

function iswpt(xw::AbstractArray{T,2}, filter::OrthoFilter, 
        ε::AbstractVector{Bool},           
        tree::BitVector) where T<:Number

    @assert isvalidtree(xw[:,1], tree)
    @assert ceil(Integer, log2(length(tree))) == length(ε)
    g, h = WT.makeqmfpair(filter)
    ss = collect(0:(length(ε)-1))
    cumsum!(ss, ε .<< ss)
    return iswpt(xw, g, h, tree, 1, ss, 1)
end

function iswpt(xw::AbstractArray{T,2}, g::Array{S,1}, h::Array{S,1}, 
        tree::BitVector, j::Integer, ss::Vector{Q}, i::Integer) where 
        {T<:Number, S<:Number, Q<:Integer}

    @assert i <= size(xw, 2)

    n₀ = length(tree)
    if i > n₀                      # bottom level leaf node
        return xw[:, i]
    elseif tree[i] == false        # leaf node
        return xw[:, i]
    end

    leftchild = iswpt(xw, g, h, tree, j+1, ss, i<<1)
    rightchild = iswpt(xw, g, h, tree, j+1, ss, i<<1+1)
    s0 = j == 1 ? 0 : ss[j-1]             # node shift
    s1 = ss[j]                            # children shift
    return isdwt_step(leftchild, rightchild, j, s0, s1, g, h)
end

# average-basis inverse stationary wavelet packet transform (ISWPT)
function iswpt(xw::AbstractArray{T,2}, wt::DiscreteWavelet,                     
        L::Integer=maxtransformlevels(size(xw,1))) where T<:Number

    return iswpt(xw, wt, maketree(size(xw,1), L, :full))
end

function iswpt(xw::AbstractArray{T,2}, filter::OrthoFilter, 
        tree::BitVector) where T<:Number
    
    @assert isvalidtree(xw[:,1], tree)
    g, h = WT.makeqmfpair(filter)
    v₀ = iswpt(xw, g, h, tree, 1, 0, 0, 1)
    v₁ = iswpt(xw, g, h, tree, 1, 0, 1, 1)
    return (v₀ + v₁) / 2
end

function iswpt(xw::AbstractArray{T,2}, g::Array{S,1}, h::Array{S,1},
        tree::BitVector, j::Integer, s0::Integer, s1::Integer, i::Integer) where    
        {T<:Number, S<:Number}

    @assert i <= size(xw, 2)

    n₀ = length(tree)
    if i > n₀                       # bottom level leaf node
        return xw[:, i]
    elseif tree[i] == false        # leaf node
        return xw[:, i]
    end

    # no additional shift
    leftchild₀ = iswpt(xw, g, h, tree, j+1, s1, s1, i<<1)
    rightchild₀ = iswpt(xw, g, h, tree, j+1, s1, s1, i<<1+1)
    v₀ = isdwt_step(leftchild₀, rightchild₀, j, s0, s1, g, h)

    # with additional shift
    leftchild₁ = iswpt(xw, g, h, tree, j+1, s1, s1 + 1<<j, i<<1)
    rightchild₁ = iswpt(xw, g, h, tree, j+1, s1, s1 + 1<<j, i<<1+1)
    v₁ = isdwt_step(leftchild₁, rightchild₁, j, s0, s1, g, h)

    return (v₀ + v₁) / 2
end

end # end module