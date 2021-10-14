# ========== ACWT Utilities ==========
"""
    autocorr(f::OrthoFilter)

Generates the autocorrelation filter for a given wavelet filter.
"""
function autocorr(f::OrthoFilter)
    H = WT.qmf(f)
    l = length(H)
    result = zeros(l - 1)
    for k in 1:(l - 1)
        for i in 1:(l - k)
            @inbounds result[k] += H[i] * H[i + k]
        end
        result[k] *= 2
    end
    return result
end

"""
    pfilter(f::OrthoFilter)

Generates the high-pass autocorrelation filter

**See also:** [`qfilter`](@ref), [`autocorr`](@ref)
"""
function pfilter(f::OrthoFilter)
    a = autocorr(f)
    c1 = 1 / sqrt(2)
    c2 = c1 / 2
    b = c2 * a
    return vcat(reverse(b), c1, b)
end

"""
    qfilter(f::OrthoFilter)

Generates the low-pass autocorrelation filter.

**See also:** [`pfilter`](@ref), [`autocorr`](@ref)
"""
function qfilter(f::OrthoFilter)
    a = autocorr(f)
    c1 = 1 / sqrt(2)
    c2 = c1 / 2
    b = -c2 * a
    return vcat(reverse(b), c1, b)
end

"""
    make_acqmfpair(f::OrthoFilter)

Generates the autocorrelation quadratic mirror filters.

**See also:** [`make_acreverseqmfpair`](@ref), [`pfilter`](@ref), [`qfilter`](@ref)
"""
function make_acqmfpair(f::OrthoFilter)
    pmfilter, qmfilter = pfilter(f), qfilter(f)
    return pmfilter, qmfilter
end

"""
    make_acreverseqmfpair(f::OrthoFilter)

Generates the reverse autocorrelation quadratic mirror filters.

**See also:** [`make_acqmfpair`](@ref), [`pfilter`](@ref), [`qfilter`](@ref)
"""
function make_acreverseqmfpair(f::OrthoFilter)
    pmf, qmf = make_acqmfpair(f)
    return reverse(pmf), reverse(qmf)
end
