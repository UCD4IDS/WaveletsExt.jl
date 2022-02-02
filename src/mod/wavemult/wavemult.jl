"""
    nonstd_wavemult(M, x, wt, [L], [ϵ])
    nonstd_wavemult(NM, x, wt, [L])

If `M` is an ``n \times n`` matrix, there are two ways to compute ``y = Mx``. The first is to use the
standard matrix multiplication to compute the product. This algorithm works in order of
``O(n^2)``, where ``n`` is the length of `x``.

The second is to transform both the matrix and vector to their nonstandard forms and
multiply the nonstandard forms. If the matrix is sparse in nonstandard form, this can be an
order ``O(n)`` algorithm.
"""
function nonstd_wavemult(M::AbstractMatrix{T},
                         x::AbstractVector{T},
                         wt::OrthoFilter,
                         L::Integer = maxtransformlevels(x),
                         ϵ::T = 1e-4) where T<:AbstractFloat
    NM = mat2sparse_nsform(M, wt, L, ϵ)
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