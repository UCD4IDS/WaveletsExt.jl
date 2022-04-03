"""
    dyadlength(x)
    dyadlength(n)

Find dyadic length of array.

# Arguments
- `x::AbstractArray`: Array of length `n`. Preferred array length is ``2^J`` where ``J`` is
  an integer.
- `n::Integer`: Length of array.

# Returns
- `J::Integer`: Least power of two greater than `n`.

!!! note
    The function `dyadlength` is very similar to the function `maxtransformlevels` from
    Wavelets.jl. The only difference here is the way it handles `n` when `n` is not a power
    of 2. The example below provides a demonstration of the differences in the 2 functions.

# Examples
```jldoctest
julia> import WaveletsExt.WaveMult: dyadlength

julia> import Wavelets: maxtransformlevels

julia> x = randn(16); dyadlength(x)
4

julia> dyadlength(16)              # Same as previous
4

julia> maxtransformlevels(16)      # Equivalent to dyadlength when n is power of 2
4

julia> dyadlength(15)              # Produces warning when n is not power of 2
┌ Warning: Dyadlength n != 2^J
└ @ WaveletsExt.WaveMult
4

julia> maxtransformlevels(15)      # Not equivalent to dyadlength when n is not power of 2
0
```
"""
function dyadlength(n::T) where T<:Integer
    J = ceil(T, log2(n))
    if 1<<J != n
        @warn "Dyadlength n != 2^J"
    end
    return J
end
dyadlength(x::AbstractArray) = dyadlength(size(x,1))

"""
    stretchmatrix(i, j, n, L)

Stretch matrix into BCR nonstandard form.

# Arguments
- `i::AbstractVector{T} where T<:Integer`: Row indices of nonzero elements of matrix.
- `j::AbstractVector{T} where T<:Integer`: Column indices of nonzero elements of matrix.
- `n::T where T<:Integer`: Size of square matrix.
- `L::T where T<:Integer`: Number of resolution levels.

# Returns
- `ie::Vector{T}`: Row indices of elements in nonstandard form of matrix.
- `je::Vector{T}`: Column indices of elements in nonstandard form of matrix.

# Examples
```jldoctest
julia> M = [1 0 0 0;
            0 2 0 0;
            0 0 3 0;
            0 0 0 4];

julia> idx = findall(!iszero, M)
2-element Vector{CartesianIndex{2}}:
 CartesianIndex(1, 1)
 CartesianIndex(2, 2)

julia> i = getindex.(idx, 1)
4-element Vector{Int64}:
 1
 2
 3
 4

julia> j = getindex.(idx, 2)
4-element Vector{Int64}:
 1
 2
 3
 4

julia> stretchmatrix(i, j, 4, 2)
([1, 4, 7, 8], [1, 4, 7, 8])
```
"""
function stretchmatrix(i::AbstractVector{T}, j::AbstractVector{T}, n::T, L::T) where 
                       T<:Integer
    Lmax = maxtransformlevels(n)
    @assert 1 ≤ L ≤ Lmax
    ie = copy(i)
    je = copy(j)
    for l in 0:(L-1)
        k = Lmax - l - 1
        cond = ((ie .> 1<<k) .| (je .> 1<<k)) .& ((ie .≤ 1<<(k+1)) .& (je .≤ 1<<(k+1)))
        idx = findall(cond)
        if !isempty(idx)
            ie[idx] = ie[idx] .+ 1<<(k+1)
            je[idx] = je[idx] .+ 1<<(k+1)
        end
    end
    return ie, je
end

"""
    ndyad(L, Lmax, gender)

Index dyad of nonstandard wavelet transform.

# Arguments
- `L::T where T<:Integer`: Current level of node.
- `Lmax::T where T<:Integer`: Max level of a signal being analyzed.
- `gender::Bool`: "Gender" of node. 
    - `true` = Female: Node is detail coefficients of the decomposition of its parent.
    - `false` = Male: Node is approximate coefficients of the decomposition of its parent.

!!! note 
    Current level of node `L` cannot be 0 or larger than its max possible level `Lmax`.

# Returns
- `::UnitRange{T}`: Range of all coefficients at the `L`-th level attached to wavelets of
  indicated gender.

# Examples
```jldoctest
julia> import WaveletsExt.WaveMult: ndyad

julia> ndyad(1, 4, false)
17:24

julia> ndyad(1, 4, true)
25:32
```
"""
function ndyad(L::T, Lmax::T, gender::Bool) where T<:Integer
    @assert L ≤ Lmax    # Current level cannot be larger than max level
    @assert L ≥ 1       # Current level must be at least 1
    k = Lmax - L
    if gender
        return (1<<(k+1) + 1<<k + 1):(1<<(k+2))
    else
        return (1<<(k+1) + 1):(1<<(k+1) + 1<<k)
    end
end