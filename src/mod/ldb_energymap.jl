import AverageShiftedHistograms: ash, xy, pdf

## ---------- ENERGY MAPS ----------
"""
Energy map for Local Discriminant Basis. Current available types are:
- [`TimeFrequency`](@ref)
- [`ProbabilityDensity`](@ref)
"""
abstract type EnergyMap end

@doc raw"""
    TimeFrequency <: EnergyMap

An energy map based on time frequencies, a measure based on the differences of 
derived quantities from projection ``Z_i``, such as mean class energies or 
cumulants.

**See also:** [`EnergyMap`](@ref), [`ProbabilityDensity`](@ref),
    [`Signatures`](@ref)
"""
struct TimeFrequency <: EnergyMap end

@doc raw"""
    ProbabilityDensity <: EnergyMap

An energy map based on probability density, a measure based on the differences 
among the pdfs of ``Z_i``. Since we do not know the true density functions of
the coefficients, the PDFs are estimated using the Average Shifted Histogram
(ASH).

**See also:** [`EnergyMap`](@ref), [`TimeFrequency`](@ref), [`Signatures`](@ref)
"""
struct ProbabilityDensity <: EnergyMap end

@doc raw"""
    Signatures <: EnergyMap

An energy map based on signatures, a measure that uses the Earth Mover's Distance (EMD) to
compute the discriminating  power of a coordinate. Signatures provide us with a fully
data-driven representation, which can be efficiently used with EMD. This representation is
more efficient than a histogram and is able to represent complex data structure with fewer
samples.

Here, a signature for the coefficients in the ``j``-th level, ``k``-th node, ``l``-th index
of class ``c`` is defined as

``s_{j,k,l}^{(c)} = \{(\alpha_{i;j,k,l}^{(c)}, w_{i;j,k,l}^{(c)})\}_{i=1}^{N_c}``

where ``\alpha_{i;j,k,l}^{(c)}`` and ``w_{i;j,k,l}^{(c)}`` are the expansion coefficients
and weights at location ``(j,k,l)`` for signal ``i`` of class ``c`` respectively. Currently,
the two valid types of weights are `:equal` and `:pdf`.

# Argumemts
- `weight::Symbol`: Type of weight to be used to compute ``w_{i;j,k,l}^{(c)}``. Available
    methods are `:equal` and `:pdf`. Default is set to `:equal`.

**See also:** [`EnergyMap`](@ref), [`TimeFrequency`](@ref), [`ProbabilityDensity`](@ref)
"""
struct Signatures <: EnergyMap 
    weight::Symbol
    # Constructor
    Signatures(weight=:equal) = weight ∈ [:equal, :pdf] ? new(weight) : 
        throw(ValueError("Invalid weight type. Valid weight types are :equal and :pdf."))
end

"""
    energy_map(Xw, y, method)

Returns the Time Frequency Energy map or the Probability Density Energy map
depending on the input `method` (`TimeFrequency()` or `ProbabilityDensity()`).

**See also:** [`EnergyMap`](@ref). [`TimeFrequency`](@ref), 
    [`ProbabilityDensity`](@ref)
"""
function energy_map(Xw::AbstractArray{S}, y::AbstractVector{T}, method::TimeFrequency) where 
                   {S<:AbstractFloat, T}
    # --- Sanity check ---
    N = ndims(Xw)
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    sz = size(Xw)[1:(N-2)]
    L = size(Xw, N-1)
    Nx = size(Xw, N)
    # parameter checking
    @assert 3 ≤ N ≤ 4
    @assert Nx == Ny
    @assert nc > 1
    @assert 1 ≤ L-1 ≤ maxtransformlevels(min(sz...))

    # --- Construct normalized energy map for each class ---
    Γ = Array{S, N}(undef, (sz...,L,nc))
    map_size = prod([sz...,L])                      # number of elements per class of energy map
    slice_dim = N==3 ? [1] : [1,2]                  # dimension for norm computation of each slice of signal
    for (i,cᵢ) in enumerate(c)
        idx = findall(yᵢ -> yᵢ==cᵢ, y)
        @inbounds x = N==3 ? Xw[:,1,idx] : Xw[:,:,1,idx]      # Set of original signals of class cᵢ
        @inbounds xw = N==3 ? Xw[:,:,idx] : Xw[:,:,:,idx]     # Set of decomposed signals of class cᵢ
        norm_sum = sum(mapslices(xᵢ -> norm(xᵢ,2)^2, x, dims = slice_dim))
        en = sum(xw.^2, dims=N)
        start_index = (i-1)*map_size                # start_index for current class in energy map
        for j in eachindex(en)
            @inbounds Γ[start_index+j] = en[j] / norm_sum
        end
    end
    return Γ
end

function energy_map(Xw::AbstractArray{S}, y::AbstractVector{T}, 
                    method::ProbabilityDensity) where {S<:AbstractFloat, T}
    # --- Sanity check ---
    N = ndims(Xw)
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    sz = size(Xw)[1:(N-2)]
    L = size(Xw, N-1)
    Nx = size(Xw, N)
    # parameter checking
    @assert 3 ≤ N ≤ 4
    @assert Nx == Ny
    @assert nc > 1
    @assert 1 ≤ L-1 ≤ maxtransformlevels(min(sz...))
    
    # --- Construct empirical probability density for each coefficent of each class ---
    nbins = ceil(Int, (30*Nx)^(1/5))    # number of bins/histogram
    mbins = ceil(Int, 100/nbins)        # number of histograms M/nbins, M=100 is arbitrary
    pdf_len = (nbins+1)*mbins           # vector length of empirical pdf
    slice_size = prod([sz...,L])        # number of elements per class of energy map slice
    map_size = prod([sz...,L,pdf_len])  # dimension for norm computation of each slice of signal
    Γ = Array{Float64,N+1}(undef, (sz..., L, pdf_len, nc))
    for (i,cᵢ) in enumerate(c)      # iterate over each class
        idx = findall(yᵢ -> yᵢ==cᵢ, y)
        @inbounds xw = N==3 ? Xw[:,:,idx] : Xw[:,:,:,idx]  # wavelet packet for class cᵢ
        for j in 1:slice_size
            @inbounds z = @view Xw[j:slice_size:end]    # Coefficients at index j for all signals
            @inbounds zᵢ = @view xw[j:slice_size:end]   # Coefficients at index j for signals in class cᵢ
            # ASH parameter setup
            σ = std(z)
            s = 0.5
            δ = (maximum(z)-minimum(z)+σ)/((nbins+1)*mbins-1)
            rng = range(minimum(z)-s*σ, step=δ, length=(nbins+1)*mbins)
            # Empirical PDF
            epdf = ash(zᵢ, rng=rng, m=mbins, kernel=Kernels.triangular)
            start_index = (i-1)*map_size    # start_index for current class in energy map
            end_index = i*map_size          # end index for current class in energy map
            @inbounds _, Γ[(start_index+j):slice_size:end_index] = xy(epdf) # copy distribution into Γ
        end
    end
    return Γ
end

function energy_map(Xw::AbstractArray{S}, y::AbstractVector{T}, method::Signatures) where 
                   {S<:AbstractFloat, T}
    # --- Sanity check ---
    N = ndims(Xw)
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    sz = size(Xw)[1:(N-2)]
    L = size(Xw, N-1)
    Nx = size(Xw, N)
    # parameter checking
    @assert 3 ≤ N ≤ 4
    @assert Nx == Ny
    @assert nc > 1
    @assert 1 ≤ L-1 ≤ maxtransformlevels(min(sz...))
    @assert method.weight ∈ [:equal, :pdf]

    # --- Form signatures in a structure of a named tuple ---
    Γ = method.weight==:equal ? 
        Vector{NamedTuple{(:coef, :weight), Tuple{Array{S}, S}}}(undef, nc) :      # equal weights
        Vector{NamedTuple{(:coef, :weight), Tuple{Array{S}, Array{S}}}}(undef, nc) # pdf-based weights
    slice_size = prod([sz...,L])        # number of elements per class of energy map slice
    for (i, cᵢ) in enumerate(c)
        idx = findall(yᵢ -> yᵢ==cᵢ, y)
        xw = N==3 ? Xw[:,:,idx] : Xw[:,:,:,idx] # wavelet packet for class cᵢ
        Nc = length(idx)                        # number of data in class cᵢ
        if method.weight == :equal
            w = 1/Nc
        else
            nbins = ceil(Int, (30*Nx)^(1/5)) # number of bins/histogram
            mbins = ceil(Int, 100/nbins)     # number of histograms M/nbins, M=100 is arbitrary

            # compute weights
            w = Array{S,N}(undef, (sz...,L,Nc))
            for j in 1:slice_size
                z = @view xw[j:slice_size:end]  # coefficients at j for cᵢ signals
                # ASH parameter setup
                σ = std(z); s = 0.5
                δ = (maximum(z)-minimum(z)+σ)/((nbins+1)*mbins-1)
                rng = range(minimum(z)-s*σ, step=δ, length=(nbins+1)*mbins)
                # Empirical PDF
                epdf = ash(z, rng=rng, m=mbins, kernel=Kernels.triangular)
                for k in 1:Nc
                    start_index = (k-1)*slice_size
                    w[start_index+j] = pdf(epdf, z[k])
                end
            end
        end
        Γ[i] = (coef = xw, weight = w)
    end
    return Γ
end
