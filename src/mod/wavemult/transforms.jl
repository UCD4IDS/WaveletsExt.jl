"""
    ns_dwt(x, wt, [L])

Nonstandard wavelet transform on 1D signals.

# Arguments
- `x::AbstractVector{T} where T<:AbstractFloat`: 1D signal of length ``2^J``.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(x)`) Number of decomposition levels.

# Returns
- `nxw::Vector{T}`: Nonstandard wavelet transform of `x` of length ``2^{J+1}``.

**See also:** [`nonstd_wavemult`](@ref), [`mat2sparseform_nonstd`](@ref), [`ns_idwt`](@ref),
[`ndyad`](@ref)
"""
function ns_dwt(x::AbstractVector{T}, 
                wt::OrthoFilter, 
                L::Integer = maxtransformlevels(x)) where T<:AbstractFloat
    Lmax = maxtransformlevels(x)
    n = length(x)
    @assert 1 ≤ L ≤ Lmax
    @assert ispow2(n)

    nxw = zeros(T, 2*n)
    g, h = WT.makereverseqmfpair(wt, true)
    for l in 1:L
        w₁ = @view nxw[ndyad(l, Lmax, false)]
        w₂ = @view nxw[ndyad(l, Lmax, true)]
        v = l == 1 ? x : @view nxw[ndyad(l-1, Lmax, false)]
        dwt_step!(w₁, w₂, v, h, g)
    end
    return nxw
end

"""
    ns_idwt(nxw, wt, [L])

Inverse nonstandard wavelet transform on 1D signals.

# Arguments
- `nxw::AbstractVector{T} where T<:AbstractFloat`: Nonstandard wavelet transformed 1D signal
  of length ``2^{J+1}``.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: (Default: `maxtransformlevels(nxw) - 1`) Number of decomposition levels.

# Returns
- `x::Vector{T}`: 1D signal of length ``2^J``.

**See also:** [`nonstd_wavemult`](@ref), [`mat2sparseform_nonstd`](@ref), [`ns_dwt`](@ref),
[`ndyad`](@ref)
"""
function ns_idwt(nxw::AbstractVector{T},
                 wt::OrthoFilter,
                 L::Integer = maxtransformlevels(nxw)-1) where T<:AbstractFloat
    Lmax = maxtransformlevels(nxw) - 1
    n = length(nxw) ÷ 2
    @assert 1 ≤ L ≤ Lmax
    @assert ispow2(n)

    x = zeros(T, n)
    x[1:1<<(Lmax-L)] = nxw[1:1<<(Lmax-L)]
    g, h = WT.makereverseqmfpair(wt, true)
    for l in L:-1:1
        w₁ = nxw[ndyad(l, Lmax, false)] + x[1:1<<(Lmax-l)] 
        w₂ = @view nxw[ndyad(l, Lmax, true)]
        v = @view x[1:1<<(Lmax-l+1)]
        idwt_step!(v, w₁, w₂, h, g)
    end
    return x
end

"""
    sft(M, wt, [L])

Transforms a matrix `M` to be then represented in the sparse standard form. This is achieved
by first computing ``L`` levels of wavelet transform on each column of `M`, and then
computing ``L`` levels of wavelet transform on each row of `M`.

# Arguments
- `M::AbstractMatrix{T} where T<:AbstractFloat`: Matrix to be put in standard form.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: Number of decomposition levels.

# Returns
- `Mw::Matrix{T}`: Matrix in the standard form.

**See also:** [`mat2sparseform_std`](@ref), [`isft`](@ref)
"""
function sft(M::AbstractMatrix{T}, 
             wt::OrthoFilter, 
             L::Integer = maxtransformlevels(M)) where T<:AbstractFloat
    @assert 1 ≤ L ≤ maxtransformlevels(M)
    n, m = size(M)

    Mw = similar(M)
    for j in 1:m        # Transform each column
        Mw[:,j] = dwt(M[:,j], wt, L)
    end
    for i in 1:n        # Transform each row
        Mw[i,:] = dwt(Mw[i,:], wt, L)
    end
    return Mw
end

"""
    isft(Mw, wt, [L])

Reconstructs the matrix `M` from the sparse standard form `Mw`. This is achieved by first
computing ``L`` levels of inverse wavelet transform on each row of `Mw`, and then computing
``L`` levels of inverse wavelet transform on each column of `Mw`.

# Arguments
- `Mw::AbstractMatrix{T} where T<:AbstractFloat`: Matrix in standard form.
- `wt::OrthoFilter`: Type of wavelet filter.
- `L::Integer`: Number of decomposition levels.

# Returns
- `M::Matrix{T}`: Reconstructed matrix.

**See also:** [`sft`](@ref)
"""
function isft(Mw::AbstractMatrix{T},
              wt::OrthoFilter,
              L::Integer = maxtransformlevels(M)) where T<:AbstractFloat
    @assert 1 ≤ L ≤ maxtransformlevels(Mw)
    n, m = size(Mw)
    
    M = similar(Mw)
    for i in 1:n        # Transform each row
        M[i,:] = idwt(Mw[i,:], wt, L)
    end
    for j in 1:m        # Transform each column
        M[:,j] = idwt(M[:,j], wt, L)
    end
    return M
end