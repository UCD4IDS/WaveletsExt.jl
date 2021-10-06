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

using ..Utils

# ========== Single Step Stationary Wavelet Transform ==========
"""
    sdwt_step(v, d, h, g)

Perform one level of the stationary discrete wavelet transform (SDWT) on the vector `v`,
which is the `d`-th level scaling coefficients (Note the 0th level scaling coefficients is the
raw signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
SWT.sdwt_step(v, 0, h, g)
```

**See also:** [`sdwt_step!`](@ref)
"""
function sdwt_step(v::AbstractVector{T}, d::Integer, h::Array{S,1}, g::Array{S,1}) where 
                  {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T, n)
    w₂ = zeros(T, n)

    sdwt_step!(w₁, w₂, v, d, h, g)
    return w₁, w₂
end

"""
    sdwt_step!(v1, w1, v, j, h, g)

Same as `sdwt_step` but without array allocation.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)
w₁ = zeros(8)
w₂ = zeros(8)

# One step of SDWT
SWT.sdwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`sdwt_step`](@ref)
"""
function sdwt_step!(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T},
                    v::AbstractVector{T},
                    d::Integer, 
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Setup
    n = length(v)                   # Signal length
    filtlen = length(h)             # Filter length

    # One step of stationary transform
    for i in 1:n
        k = i
        @inbounds w₁[i] = g[1] * v[k]
        @inbounds w₂[i] = h[1] * v[k]
        for j in 2:filtlen
            k += 1 << d
            k = k > n ? mod1(k,n) : k
            @inbounds w₁[i] += g[j] * v[k]
            @inbounds w₂[i] += h[j] * v[k]
        end
    end
    return w₁, w₂
end

"""
    isdwt_step(w₁, w₂, d, h, g)
    isdwt_step(w₁, w₂, d, sv, sw, h, g)

Perform one level of the inverse stationary discrete wavelet transform (ISDWT) on the
vectors `w₁` and `w₂`, which are the `d+1`-th level scaling coefficients (Note the 0th level
scaling coefficients is the raw signal). The vectors `h` and `g` are the detail and scaling
filters.

!!! note
    One can decide to choose the average-based inverse transform or the shift-based inverse
    transform. For shift based, one needs to specify the shifts of the parent and children
    nodes; for the average based, the output is the average of all possible shift-based
    inverse transform outputs.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
v̂ = SWT.isdwt_step(w₁, w₂, 0, h, g)         # Average based
ṽ = SWT.isdwt_step(w₁, w₂, 0, 0, 1, h, g)   # Shift based
```

**See also:** [`isdwt_step!`](@ref)
"""
function isdwt_step(w₁::AbstractVector{T},
                    w₂::AbstractVector{T},
                    d::Integer,
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Setup
    v₁ = similar(w₁)        # Allocation for isdwt_step of w₁, w₂ with no additional shift
    v₂ = similar(w₁)        # Allocation for isdwt_step of w₁, w₂ with additional shift
    n = length(w₁)          # Signal length
    nd = 1 << d             # Number of blocks for parent node

    # isdwt_step for each shift
    for sv in 0:(nd-1)
        sw₁ = sv            # Shift of w₁, w₂ with no additional shift
        sw₂ = sv + 1<<d     # Shift of w₁, w₂ with addtional shift
        isdwt_step!(v₁, w₁, w₂, d, sv, sw₁, h, g)   # Without shift
        isdwt_step!(v₂, w₁, w₂, d, sv, sw₂, h, g)   # With shift
    end
    return (v₁+v₂)/2        # Average the results of v₁, v₂
end

function isdwt_step(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T}, 
                    d::Integer, 
                    sv::Integer, 
                    sw::Integer, 
                    h::Array{S,1}, 
                    g::Array{S,1}) where {T<:Number, S<:Number}
    v = similar(w₁)
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
    return v
end

"""
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)

Same as `isdwt_step` but without array allocation.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
v̂ = similar(v)
ṽ = similar(v)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
SWT.isdwt_step(v̂, w₁, w₂, 0, h, g)          # Average based
SWT.isdwt_step(ṽ, w₁, w₂, 0, 0, 1, h, g)    # Shift based
```

**See also:** [`isdwt_step`](@ref)
"""
function isdwt_step!(v::AbstractVector{T}, 
                     w₁::AbstractVector{T}, 
                     w₂::AbstractVector{T}, 
                     d::Integer, 
                     sv::Integer, 
                     sw::Integer,
                     h::Array{S,1}, 
                     g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert sw ≥ sv

    # Setup
    n = length(v)               # Signal length
    filtlen = length(h)         # filter length
    ip = sv + 1                 # Parent start index
    sp = 1 << d                 # Parent step size
    ic = sw + 1                 # Child start index
    sc = 1 << (d+1)             # Child step size
    
    # One step of inverse SDWT
    for (t, m) in enumerate(ip:sp:n)
        i = mod1(t,2)
        j = sw == sv ? m : mod1(m+sp, n)        # Circshift needed if sw > sv
        k = (ceil(Int, t/2) - 1) * sc + ic      # Child start index for calculating v[j]
        v[j] = h[i] * w₂[k] + g[i] * w₁[k]      # Calculation of v[j]
        while i + 2 ≤ filtlen
            k -= 1 << (d+1)
            k = k ≤ 0 ? mod1(k,n) : k
            i += 2
            v[j] += h[i] * w₂[k] + g[i] * w₁[k]
        end
    end
    return v
end

# ========== Stationary DWT ==========
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
function sdwt(x::AbstractVector{T}, wt::OrthoFilter, 
        L::Integer=maxtransformlevels(x)) where T<:Number
    
    @assert L <= maxtransformlevels(x) ||
        throw(ArgumentError("Too many transform levels (length(x) < 2^L"))
    @assert L >= 1 || throw(ArgumentError("L must be >= 1"))

    g, h = WT.makereverseqmfpair(wt, true)
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

# ε-basis inverse stationary discrete wavelet transform (ISDWT)
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

# ========== Stationary WPT ==========
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


# ========== Stationary WPD ==========
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

# ε-basis inverse stationary wavelet packet transform (ISWPT)
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