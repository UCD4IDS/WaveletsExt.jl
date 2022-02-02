function ns_dwt(x::AbstractVector{T}, 
                wt::OrthoFilter, 
                L::Integer = maxtransformlevels(x)) where T<:AbstractFloat
    Lmax = maxtransformlevels(x)
    @assert L ≤ Lmax
    n = length(x)
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

"""
function ns_idwt(nxw::AbstractVector{T},
                 wt::OrthoFilter,
                 L::Integer = maxtransformlevels(nxw)-1) where T<:AbstractFloat
    Lmax = maxtransformlevels(nxw) - 1
    @assert L ≤ Lmax
    n = length(nxw) ÷ 2
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