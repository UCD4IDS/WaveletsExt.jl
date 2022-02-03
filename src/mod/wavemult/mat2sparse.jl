"""
    mat2sparseform_nonstd(M, wt, [L], [ϵ])

Transform the matrix `M` into the wavelet basis. Then, it is stretched into its nonstandard
form. Elements exceeding `ϵ * maximum column norm` are set to zero. The resulting output
sparse matrix, and can be used as input to `nonstd_wavemult`.

# Arguments
- `M::AbstractMatrix{T} where T<:AbstractFloat`: `n` by `n` matrix (`n` dyadic) to be put in
  Sparse Nonstandard form. 
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(M)`) Number of decomposition levels.
- `ϵ::T where T<:AbstractFloat`: (Default: `1e-4`) Truncation Criterion.

# Returns
- `NM::SparseMatrixCSC{T, Integer}`: Sparse nonstandard form of matrix of size 2n x 2n.

**See also:** [`mat2sparseform_std`](@ref), [`stretchmatrix`](@ref)
"""
function mat2sparseform_nonstd(M::AbstractMatrix{T}, 
                               wt::OrthoFilter,
                               L::Integer = maxtransformlevels(M),
                               ϵ::T = 1e-4) where T<:AbstractFloat
    @assert size(M,1) == size(M,2)
    n = size(M, 1)
    Mw = dwt(M, wt, L)                          # 2D wavelet transform
    # Find maximum column norm of Mw
    maxcolnorm = (maximum ∘ mapslices)(norm, Mw, dims = 1)
    nilMw = Mw .* (abs.(Mw) .> ϵ * maxcolnorm)  # "Remove" values close to zero
    nz_vals = nilMw[nilMw .≠ 0]                 # Extract all nonzero values
    nz_indx = findall(!iszero, nilMw)           # Extract indices of nonzero values
    i = getindex.(nz_indx, 1)
    j = getindex.(nz_indx, 2)
    ie, je = stretchmatrix(i, j, n, L)          # Stretch matrix Mw
    NM = sparse(ie, je, nz_vals, 2*n, 2*n)      # Convert to sparse matrix
    return NM
end

"""
    mat2sparseform_std(M, wt, [L], [ϵ])

Transform the matrix `M` into the standard form. Then, elements exceeding `ϵ * maximum
column norm` are set to zero. The resulting output sparse matrix, and can be used as input
to `std_wavemult`.

# Arguments
- `M::AbstractMatrix{T} where T<:AbstractFloat`: Matrix to be put in Sparse Standard form. 
- `wt::OrthoFilter`: Wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(M)`) Number of decomposition levels.
- `ϵ::T where T<:AbstractFloat`: (Default: `1e-4`) Truncation Criterion.

# Returns
- `SM::SparseMatrixCSC{T, Integer}`: Sparse standard form of matrix of size ``n \times n``.

**See also:** [`mat2sparseform_nonstd`](@ref), [`sft`](@ref)
"""
function mat2sparseform_std(M::AbstractMatrix{T},
                            wt::OrthoFilter,
                            L::Integer = maxtransformlevels(M),
                            ϵ::T = 1e-4) where T<:AbstractFloat
    @assert size(M,1) == size(M,2)
    Mw = sft(M, wt, L)                          # Transform to standard form
    # Find maximum column norm of Mw
    maxcolnorm = (maximum ∘ mapslices)(norm, Mw, dims = 1)
    nilMw = Mw .* (abs.(Mw) .> ϵ * maxcolnorm)  # Remove values close to zero
    SM = sparse(nilMw)                          # Convert to sparse matrix
    return SM
end