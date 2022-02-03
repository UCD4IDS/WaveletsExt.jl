"""
    nonstd_wavemult(M, x, wt, [L], [ϵ])
    nonstd_wavemult(NM, x, wt, [L])

If `M` is an ``n \times n`` matrix, there are two ways to compute ``y = Mx``. The first is
to use the standard matrix multiplication to compute the product. This algorithm works in
order of ``O(n^2)``, where ``n`` is the length of `x``.

The second is to transform both the matrix and vector to their nonstandard forms and
multiply the nonstandard forms. If the matrix is sparse in nonstandard form, this can be an
order ``O(n)`` algorithm.

!!! tip
    One may choose to use the original matrix `M` as input by doing `nonstd_wavemult(M, x,
    wt, [L], [ϵ])`. However, if the nonstandard form sparse matrix `NM` is already computed
    prior, one can skip the redundant step by doing `nonstd_wavemult(NM, x, wt, [L])`.

# Arguments
- `M::AbstractVector{T} where T<:AbstractFloat`: ``n \times n`` matrix.
- `NM::SparseMatrixCSC{T,S} where {T<:AbstractFloat, S<:Integer}`: Nonstandard transformed
  sparse matrix of `M`.
- `x::AbstractVector{T} where T<:AbstractFloat`: Vector of length ``n`` in natural basis.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of decomposition levels.
- `ϵ::T where T<:AbstractFloat`: (Default: `1e-4`) Truncation criterion for nonstandard
  transform of `M`.

# Returns
- `y::Vector{T}`: Standard approximation of ``Mx``.
"""
function nonstd_wavemult(M::AbstractMatrix{T},
                         x::AbstractVector{T},
                         wt::OrthoFilter,
                         L::Integer = maxtransformlevels(x),
                         ϵ::T = 1e-4) where T<:AbstractFloat
    NM = mat2sparseform_nonstd(M, wt, L, ϵ)
    return nonstd_wavemult(NM, x, wt, L)
end

function nonstd_wavemult(NM::SparseMatrixCSC{T,S},
                         x::AbstractVector{T},
                         wt::OrthoFilter,
                         L::Integer = maxtransformlevels(x)) where 
                        {T<:AbstractFloat, S<:Integer}
    nx = ns_dwt(x, wt, L)
    ny = NM * nx
    y = ns_idwt(ny, wt, L)
    return y
end

"""
    std_wavemult(M, x, wt, [L], [ϵ])
    std_wavemult(SM, x, wt, [L])

If `M` is an ``n \times n`` matrix, there are two ways to compute ``y = Mx``. The first is
to use the standard matrix multiplication to compute the product. This algorithm works in
order of ``O(n^2)``, where ``n`` is the length of `x``.

The second is to transform both the matrix and vector to their standard forms and multiply
the standard forms. If the matrix is sparse in standard form, this can be an order ``O(n)``
algorithm.

!!! tip 
    One may choose to use the original matrix `M` as input by doing `std_wavemult(M, x,
    wt, [L], [ϵ])`. However, if the standard form sparse matrix `SM` is already computed
    prior, one can skip the redundant step by doing `std_wavemult(SM, x, wt, [L])`.

# Arguments
- `M::AbstractVector{T} where T<:AbstractFloat`: ``n \times n`` matrix.
- `NM::SparseMatrixCSC{T,S} where {T<:AbstractFloat, S<:Integer}`: Standard transformed
  sparse matrix of `M`.
- `x::AbstractVector{T} where T<:AbstractFloat`: Vector of length ``n`` in natural basis.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of decomposition levels.
- `ϵ::T where T<:AbstractFloat`: (Default: `1e-4`) Truncation criterion for standard
  transform of `M`.

# Returns
- `y::Vector{T}`: Standard form approximation of ``Mx``.
"""
function std_wavemult(M::AbstractMatrix{T},
                      x::AbstractVector{T},
                      wt::OrthoFilter,
                      L::Integer = maxtransformlevels(x),
                      ϵ::T = 1e-4) where T<:AbstractFloat
    SM = mat2sparseform_std(M, wt, L, ϵ)
    return std_wavemult(SM, x, wt, L)
end

function std_wavemult(SM::SparseMatrixCSC{T,S},
                      x::AbstractVector{T},
                      wt::OrthoFilter,
                      L::Integer = maxtransformlevels(x)) where 
                     {T<:AbstractFloat, S<:Integer}
    nx = dwt(x, wt, L)
    ny = SM * nx
    y = idwt(ny, wt, L)
    return y
end