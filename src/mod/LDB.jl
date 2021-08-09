module LDB
using Base: AbstractFloat
export 
    # energy map
    EnergyMap,
    TimeFrequency,
    ProbabilityDensity,
    Signatures,
    energy_map,
    # discriminant measures
    DiscriminantMeasure,
    ProbabilityDensityDM,
    SignaturesDM,
    AsymmetricRelativeEntropy,
    SymmetricRelativeEntropy,
    HellingerDistance,
    LpDistance,
    EarthMoverDistance,
    discriminant_measure,
    # discriminant power
    DiscriminantPower,
    BasisDiscriminantMeasure,
    FishersClassSeparability,
    RobustFishersClassSeparability,
    discriminant_power,
    # local discriminant basis
    LocalDiscriminantBasis,
    fit!,
    fit_transform,
    transform,
    inverse_transform,
    change_nfeatures

using
    AverageShiftedHistograms,
    LinearAlgebra,
    Wavelets,
    Parameters,
    Statistics,
    StatsBase

using 
    ..Utils,
    ..WPD,
    ..BestBasis



## ENERGY MAPS
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

An energy map based on signatures, a measure that uses the Earth Mover's
Distance (EMD) to compute the discriminating  power of a coordinate. Signatures
provide us with a fully data-driven representation, which can be efficiently
used with EMD. This representation is more efficient than a histogram and is
able to represent complex data structure with fewer samples.

Here, a signature for the coefficients in the ``j``-th level, ``k``-th node,
``l``-th index of class ``c`` is defined as

``s_{j,k,l}^{(c)} = \{(\alpha_{i;j,k,l}^{(c)}, w_{i;j,k,l}^{(c)})\}_{i=1}^{N_c}``

where ``\alpha_{i;j,k,l}^{(c)}`` and ``w_{i;j,k,l}^{(c)}`` are the expansion 
coefficients and weights at location ``(j,k,l)`` for signal ``i`` of class ``c``
respectively. Currently, the two valid types of weights are `:equal` and `:pdf`.

# Argumemts
- `weight::Symbol`: Type of weight to be used to compute ``w_{i;j,k,l}^{(c)}``.
    Available methods are `:equal` and `pdf`. Default is set to `:equal`.

**See also:** [`EnergyMap`](@ref), [`TimeFrequency`](@ref),
    [`ProbabilityDensity`](@ref)
"""
struct Signatures <: EnergyMap 
    weight::Symbol
    Signatures(weight=:equal) = weight ∈ [:equal, :pdf] ? new(weight) : 
        error("Invalid weight type. Valid weight types are :equal and :pdf.")
end

"""
    energy_map(Xw, y, method)

Returns the Time Frequency Energy map or the Probability Density Energy map
depending on the input `method` (`TimeFrequency()` or `ProbabilityDensity()`).

**See also:** [`EnergyMap`](@ref). [`TimeFrequency`](@ref), 
    [`ProbabilityDensity`](@ref)
"""
function energy_map(Xw::AbstractArray{S,3}, y::AbstractVector{T}, 
        method::TimeFrequency) where {S<:Number, T}

    # basic summary of data
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    n, L, Nx = size(Xw)
    
    # parameter checking
    @assert Nx == Ny
    @assert nc > 1
    @assert isdyadic(n)
    @assert 1 <= L-1 <= maxtransformlevels(n)

    # construct normalized energy map for each class
    Γ = Array{Float64, 3}(undef, (n,L,nc))
    @inbounds begin
        for (i,cᵢ) in enumerate(c)
            idx = findall(yᵢ -> yᵢ==cᵢ, y)
            x = Xw[:,1,idx]
            norm_sum = sum(mapslices(xᵢ -> norm(xᵢ,2)^2, x, dims = 1))
            en = sum(Xw[:,:,idx].^2, dims=3)
            Γ[:,:,i] = en / norm_sum
        end
    end
    
    return Γ
end

function energy_map(Xw::AbstractArray{S,3}, y::AbstractVector{T},
        method::ProbabilityDensity) where {S<:Number, T}

    # basic summary of data
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    n, L, Nx = size(Xw)
    
    # parameter checking
    @assert Nx == Ny
    @assert nc > 1
    @assert isdyadic(n)
    @assert 1 <= L-1 <= maxtransformlevels(n)

    # construct empirical probability density for each coefficent of each class
    nbins = ceil(Int, (30*Nx)^(1/5)) # number of bins/histogram
    mbins = ceil(Int, 100/nbins)     # number of histograms M/nbins, M=100 is arbitrary
    Γ = Array{Float64,4}(undef, (n, L, (nbins+1)*mbins, nc))
    @inbounds begin
        for (i,cᵢ) in enumerate(c)      # iterate over each class
            idx = findall(yᵢ -> yᵢ==cᵢ, y)
            xw = Xw[:,:,idx]            # wavelet packet for class cᵢ
            for j in axes(xw,1)
                for k in axes(xw,2)
                    z = @view Xw[j,k,:] # coefficients at (j,k) for all signals
                    zᵢ = xw[j,k,:]      # coefficients at (j,k) for cᵢ signals
                    
                    # ash parameter setup
                    σ = std(z)
                    s = 0.5
                    δ = (maximum(z)-minimum(z)+σ)/((nbins+1)*mbins-1)
                    rng = range(minimum(z)-s*σ, step=δ, length=(nbins+1)*mbins)
                
                    # empirical pdf
                    epdf = ash(zᵢ, rng=rng, m=mbins, kernel=Kernels.triangular)
                    _, Γ[j,k,:,i] = xy(epdf)
                end
            end
        end
    end

    return Γ
end

function energy_map(Xw::AbstractArray{S,3}, y::AbstractVector{T},
        method::Signatures) where {S<:Number, T}

    # basic summary of data
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    n, L, Nx = size(Xw)
    
    # parameter checking
    @assert Nx == Ny
    @assert nc > 1
    @assert isdyadic(n)
    @assert 1 <= L-1 <= maxtransformlevels(n)

    # form signatures in a structure of a named tuple
    Γ = method.weight==:equal ? 
        Array{NamedTuple{(:coef, :weight), Tuple{Array{S}, Float64}},1}(undef, nc) :      # equal weights
        Array{NamedTuple{(:coef, :weight), Tuple{Array{S}, Array{Float64}}},1}(undef, nc) # pdf-based weights
    for (i, cᵢ) in enumerate(c)
        idx = findall(yᵢ -> yᵢ==cᵢ, y)
        xw = Xw[:,:,idx]            # wavelet packet for class cᵢ
        if method.weight == :equal
            Nc = length(idx)
            w = 1/Nc
        else
            Nc = length(idx)
            nbins = ceil(Int, (30*Nx)^(1/5)) # number of bins/histogram
            mbins = ceil(Int, 100/nbins)     # number of histograms M/nbins, M=100 is arbitrary

            # compute weights
            w = Array{Float64,3}(undef, (n,L,Nc))
            for j in axes(xw,1)
                for k in axes(xw,2)
                    z = @view xw[j,k,:] # coefficients at (j,k) for cᵢ signals
                    
                    # ash parameter setup
                    σ = std(z)
                    s = 0.5
                    δ = (maximum(z)-minimum(z)+σ)/((nbins+1)*mbins-1)
                    rng = range(minimum(z)-s*σ, step=δ, length=(nbins+1)*mbins)
                
                    # empirical pdf
                    epdf = ash(z, rng=rng, m=mbins, kernel=Kernels.triangular)
                    for l in 1:Nc
                        w[j,k,l] = AverageShiftedHistograms.pdf(epdf, z[l])
                    end
                end
            end
        end
        Γ[i] = (coef = xw, weight = w)
    end
    return Γ
end

## DISCRIMINANT MEASURES
"""
Discriminant measure for Local Discriminant Basis. Current available subtypes
are:
- [`ProbabilityDensityDM`](@ref)
- [`SignaturesDM`](@ref)
"""
abstract type DiscriminantMeasure end

"""
Discriminant measure for Probability Density and Time Frequency based energy 
maps. Current available measures are:
- [`AsymmetricRelativeEntropy`](@ref)
- [`SymmetricRelativeEntropy`](@ref)
- [`LpDistance`](@ref)
- [`HellingerDistance`](@ref)
"""
abstract type ProbabilityDensityDM <: DiscriminantMeasure end

"""
Discriminant measure for Signatures based energy maps. Current available
measures are:
- [`EarthMoverDistance`](@ref)
"""
abstract type SignaturesDM <: DiscriminantMeasure end

@doc raw"""
    AsymmetricRelativeEntropy <: ProbabilityDensityDM

Asymmetric Relative Entropy discriminant measure for the Probability Density and
Time Frequency based energy maps. This measure is also known as cross entropy 
and Kullback-Leibler divergence.

Equation: ``D(p,q) = \sum p(x) \log \frac{p(x)}{q(x)}``
"""
struct AsymmetricRelativeEntropy <: ProbabilityDensityDM end

@doc raw"""
    SymmetricRelativeEntropy <: ProbabilityDensityDM

Symmetric Relative Entropy discriminant measure for the Probability Density and 
Time Frequency energy maps. Similar idea to the Asymmetric Relative Entropy, but 
this aims to make the measure more symmetric.

Equation: Denote the Asymmetric Relative Entropy as ``D_A(p,q)``, then

``D(p,q) = D_A(p,q) + D_A(q,p) = \sum p(x) \log \frac{p(x)}{q(x)} + q(x) \log \frac{q(x)}{p(x)}``
"""
struct SymmetricRelativeEntropy <: ProbabilityDensityDM end

@doc raw"""
    LpDistance <: ProbabilityDensityDM

``\ell^p`` Distance discriminant measure for the Probability Density and Time 
Frequency based energy maps. The default ``p`` value is set to 2.

Equation: ``W(q,r) = ||q-r||_p^p = \sum_{i=1}^n (q_i - r_i)^p``
"""
@with_kw struct LpDistance <: ProbabilityDensityDM 
    p::Number = 2
end

@doc raw"""
    HellingerDistance <: ProbabilityDensityDM

Hellinger Distance discriminant measure for the Probability Density energy 
map.

Equation: ``H(p,q) = \sum_{i=1}^n (\sqrt{p_i} - \sqrt{q_i})^2``
"""
struct HellingerDistance <: ProbabilityDensityDM end

@doc raw"""
    EarthMoverDistance <: SignaturesDM

Earth Mover Distance discriminant measure for the Signatures energy map.

Equation: 
``E(P,Q) = \frac{\sum_{k=1}^{m+n+1} |\hat p_k - \hat q_k| (r_{k+1} - r_k)}{w_\Sigma}``

where ``r_1, r_2, \ldots, r_{m+n}`` is the sorted list of ``p_1, \ldots, p_m, 
q_1, \ldots, q_n`` and ``\hat p_k = \sum_{p_i \leq r_k} w_{p_i}``, 
``\hat q_k = \sum_{q)i \leq r_k} w_{q_i}``.
"""
struct EarthMoverDistance <: SignaturesDM end

"""
    discriminant_measure(Γ, dm)

Returns the discriminant measure of each node calculated from the energy maps.
"""
function discriminant_measure(Γ::AbstractArray{T}, 
        dm::ProbabilityDensityDM) where T<:Number

    # basic summary of data
    @assert 3 <= ndims(Γ) <= 4
    n = size(Γ,1)
    L = size(Γ,2)
    C = size(Γ)[end]
    @assert C > 1       # ensure more than 1 class

    D = zeros(Float64, (n,L))
    @inbounds begin
        for i in 1:(C-1)
            for j in (i+1):C
                if ndims(Γ) == 3
                    D += discriminant_measure(Γ[:,:,i], Γ[:,:,j], dm)
                else
                    D += discriminant_measure(Γ[:,:,:,i], Γ[:,:,:,j], dm)
                end
            end
        end
    end

    return D
end

# discriminant measure for EMD
function discriminant_measure(
        Γ::AbstractArray{NamedTuple{(:coef, :weight), Tuple{S1,S2}},1},
        dm::SignaturesDM) where {S1<:Array{<:Number}, 
        S2<:Union{AbstractFloat,Array{<:AbstractFloat}}}

    # basic summary of data
    C = length(Γ)
    @assert C > 1
    n = size(Γ[1][:coef], 1)
    L = size(Γ[1][:coef], 2)

    D = zeros(Float64, (n,L))
    @inbounds begin
        for i in 1:(C-1)
            for j in (i+1):C
                D += discriminant_measure(Γ[i], Γ[j], dm)
            end
        end
    end

    return D
end

# discriminant measure between 2 energy maps
function discriminant_measure(Γ₁::AbstractArray{T}, Γ₂::AbstractArray{T}, 
        dm::ProbabilityDensityDM) where T<:Number

    # parameter checking and basic summary
    @assert 2 <= ndims(Γ₁) <= 3
    @assert size(Γ₁) == size(Γ₂)
    n = size(Γ₁,1)
    L = size(Γ₁,2)

    D = Array{T, 2}(undef, (n,L))
    @inbounds begin
        for i in axes(D,1)
            for j in axes(D,2)
                if ndims(Γ₁) == 2       # time frequency energy map case
                    D[i,j] = discriminant_measure(Γ₁[i,j], Γ₂[i,j], dm)
                else                    # probability density energy map case
                    for k in axes(Γ₁,3)
                        D[i,j] += discriminant_measure(Γ₁[i,j,k], Γ₂[i,j,k], dm)
                    end
                end
            end
        end
    end

    return D
end

# discriminant measure between 2 nergy maps for EMD
function discriminant_measure(Γ₁::NamedTuple{(:coef, :weight), Tuple{S1,S2}}, 
        Γ₂::NamedTuple{(:coef, :weight), Tuple{S1,S2}}, 
        dm::SignaturesDM) where
        {S1<:Array{T} where T<:Number, 
        S2<:Union{AbstractFloat,Array{<:AbstractFloat}}}

    # parameter checking and basic summary
    n = size(Γ₁[:coef],1)
    L = size(Γ₁[:coef],2)
    @assert n == size(Γ₁[:coef],1) == size(Γ₂[:coef],1)
    @assert L == size(Γ₁[:coef],2) == size(Γ₂[:coef],2)

    D = Array{Float64,2}(undef, (n,L))
    for i in 1:n
        for j in 1:L
            # signatures
            if typeof(Γ₁[:weight]) <: AbstractFloat # equal weight
                P = (coef=Γ₁[:coef][i,j,:], weight=Γ₁[:weight])
                Q = (coef=Γ₂[:coef][i,j,:], weight=Γ₂[:weight])
            else                                    # probability density weight
                P = (coef=Γ₁[:coef][i,j,:], weight=Γ₁[:weight][i,j,:])
                Q = (coef=Γ₂[:coef][i,j,:], weight=Γ₂[:weight][i,j,:])
            end
            D[i,j] = discriminant_measure(P, Q, dm)
        end
    end
    return D
end

# Asymmetric Relative Entropy
function discriminant_measure(p::T, q::T, dm::AsymmetricRelativeEntropy) where 
        T<:Number

    # parameter checking
    @assert p >= 0 && q >= 0

    if p == 0 || q == 0
        return 0
    else
        return p * log(p/q)
    end
end

# Symmetric Relative Entropy
function discriminant_measure(p::T, q::T, dm::SymmetricRelativeEntropy) where 
        T<:Number

    return discriminant_measure(p, q, AsymmetricRelativeEntropy()) + 
        discriminant_measure(q, p, AsymmetricRelativeEntropy())
end

# Hellinger Distance
function discriminant_measure(p::T, q::T, dm::HellingerDistance) where T<:Number
    return (sqrt(p) - sqrt(q))^2
end

# Lᵖ Distance
function discriminant_measure(p::T, q::T, dm::LpDistance) where T<:Number
    return (p - q)^dm.p
end

# Earth Mover Distance
function discriminant_measure(P::NamedTuple{(:coef, :weight), Tuple{S1,S2}}, 
        Q::NamedTuple{(:coef, :weight), Tuple{S1, S2}}, 
        dm::EarthMoverDistance) where
        {S1<:Vector{T} where T<:Number, 
        S2<:Union{AbstractFloat,Vector{<:AbstractFloat}}}

    # assigning tuple signatures into coef and weight
    p, w_p = P
    q, w_q = Q

    # sort signature values
    p_order = sortperm(p)
    p = p[p_order]
    w_p = typeof(w_p)<:AbstractFloat ? repeat([w_p], length(p)) : w_p[p_order]
    q_order = sortperm(q)
    q = q[q_order]
    w_q = typeof(w_q)<:AbstractFloat ? repeat([w_q], length(q)) : w_q[q_order]

    # concatenate p and q, then sort them
    r = [p; q]
    sort!(r)

    # compute emd
    n = length(r)
    emd = 0
    for i in 1:(n-1)
        # get total weight of p and q less than or equal to r[i]
        p_less = p .<= r[i]
        ∑w_p = sum(w_p[p_less])
        q_less = q .<= r[i]
        ∑w_q = sum(w_q[q_less])
        # add into emd
        emd += abs(∑w_p - ∑w_q) * (r[i+1] - r[i])
    end
    emd /= (sum(w_p) + sum(w_q))
    return emd
end

## DISCRIMINATION POWER
"""
Discriminant Power measure for the Local Discriminant Basis. Current available
measures are
- [`BasisDiscriminantMeasure`](@ref)
- [`FishersClassSeparability`](@ref)
- [`RobustFishersClassSeparability`](@ref)
"""
abstract type DiscriminantPower end

"""
    BasisDiscriminantMeasure <: DiscriminantPower

This is the discriminant measure of a single basis function computed in a 
previous step to construct the energy maps.
"""
struct BasisDiscriminantMeasure <: DiscriminantPower end

@doc raw"""
    FishersClassSeparability <: DiscriminantPower

The Fisher's class separability of the expansion coefficients in the basis 
function.

Equation: ``\frac{\sum_{c=1}^C \pi_c({\rm mean}_i(\alpha_{\lambda,i}^{(c)}) - {\rm mean}_c({\rm mean}_i(\alpha_{\lambda,i}^{(c)})))^2}{\sum_{c=1}^C \pi_c {\rm var}_i(\alpha_{\lambda,i}^{(c)})}``
"""
struct FishersClassSeparability <: DiscriminantPower end

@doc raw"""
    RobustFishersClassSeparability <: DiscriminantPower

The robust version of Fisher's class separability of the expansion coefficients 
in the basis function.

Equation: ``\frac{\sum_{c=1}^C \pi_c({\rm med}_i(\alpha_{\lambda,i}^{(c)}) - {\rm med}_c({\rm med}_i(\alpha_{\lambda,i}^{(c)})))^2}{\sum_{c=1}^C \pi_c {\rm mad}_i(\alpha_{\lambda,i}^{(c)})}``
"""
struct RobustFishersClassSeparability <: DiscriminantPower end

"""
    discriminant_power(D, tree, dp)

Returns the discriminant power of each leaf from the local discriminant basis
(LDB) tree. 
"""
function discriminant_power(D::AbstractArray{T,2}, tree::BitVector, 
        dp::BasisDiscriminantMeasure) where T<:Number

    @assert length(tree) == size(D,1) - 1

    power = bestbasiscoef(D, tree)
    order = sortperm(power, rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T,2}, y::AbstractVector{S}, 
        dp::FishersClassSeparability) where {T<:Number, S}

    n = size(coefs,1)                             # signal length
    
    classes = unique(y)
    C = length(coefs)                             # number of classes
    
    Nᵢ = Array{T,1}(undef, C)
    Eαᵢ = Array{T,2}(undef, (n,C))                # mean of each entry
    Varαᵢ = Array{T,2}(undef, (n,C))              # variance of each entry
    @inbounds begin
        for (i, c) in enumerate(classes)
            idx = findall(yᵢ -> yᵢ == c, y)
            Nᵢ[i] = length(idx)
            Eαᵢ[:,i] = mean(coefs[:, idx], dims = 2)
            Varαᵢ[:,i] = var(coefs[:, idx], dims = 2)
        end
    end
    Eα = mean(Eαᵢ, dims = 2)                      # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                             # proportions of each class

    power = ((Eαᵢ - (Eα .* Eαᵢ)).^2 * pᵢ) ./ (Varαᵢ * pᵢ)
    order = sortperm(power, rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T,2}, y::AbstractVector{S},
        dp::RobustFishersClassSeparability) where {T<:Number,S}

    n = size(coefs,1)                            # signal length
    
    classes = unique(y)
    C = length(classes)

    Nᵢ = Array{T,1}(undef, C)
    Medαᵢ = Array{T,2}(undef, (n,C))             # mean of each entry
    Madαᵢ = Array{T,2}(undef, (n,C))             # variance of each entry
    @inbounds begin
        for (i, c) in enumerate(classes)
            idx = findall(yᵢ -> yᵢ == c, y)
            Nᵢ[i] = length(idx)
            Medαᵢ[:,i] = median(coefs[:, idx], dims = 2)
            Madαᵢ[:,i] = mapslices(x -> mad(x, normalize = false), coefs[:, idx], 
                dims = 2)
        end
    end
    Medα = median(Medαᵢ, dims = 2)               # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                            # proportions of each class

    power = ((Medαᵢ - (Medα .* Medαᵢ)).^2 * pᵢ) ./ (Madαᵢ * pᵢ)
    order = sortperm(power, rev = true)

    return (power, order)
end

## LOCAL DISCRIMINANT BASIS
"""
    LocalDiscriminantBasis

Class type for the Local Discriminant Basis (LDB), a feature selection algorithm
developed by N. Saito and R. Coifman in "Local Discriminant Bases and Their
Applications" in the Journal of Mathematical Imaging and Vision, Vol 5, 337-358
(1995). This struct contains the following field values: 

# Parameters and Attributes:
- `wt::DiscreteWavelet`: a discrete wavelet for transform purposes
- `max_dec_level::Union{Integer, Nothing}`: max level of wavelet packet
    decomposition to be computed.
- `dm::DiscriminantMeasure`: the discriminant measure for the LDB algorithm. 
    Supported measures are the `AsymmetricRelativeEntropy()`, `LpDistance()`,
    `SymmetricRelativeEntropy()`, and `HellingerDistance()`
- `en::EnergyMap`: the type of energy map used. Supported maps are 
    `TimeFrequency()`, `ProbabilityDensity()`, and `Signatures()`.
- `dp::DiscriminantPower()`: the measure of discriminant power among expansion
    coefficients. Supported measures are `BasisDiscriminantMeasure()`,
    `FishersClassSeparability()`, and `RobustFishersClassSeparability()`. 
- `top_k::Union{Integer, Nothing}`: the top-k coefficients used in each node to 
    determine the discriminant measure.
- `n_features::Union{Integer, Nothing}`: the dimension of output after 
    undergoing feature selection and transformation.
- `n::Union{Integer, Nothing}`: length of signal
- `Γ::Union{AbstractArray{<:AbstractFloat}, 
AbstractArray{NamedTuple{(:coef, :weight), Tuple{S1, S2}}} where {S1<:Array{T} 
    where T<:AbstractFloat, S2<:Union{AbstractFloat, Array{<:AbstractFloat}}},
    Nothing}`: computed energy map
- `DM::Union{AbstractArray{<:AbstractFloat}, Nothing}`: computed discriminant
    measure
- `cost::Union{AbstractVector{<:AbstractFloat}, Nothing}`: computed wavelet
    packet decomposition (WPD) tree cost based on the discriminant measure `DM`.
- `tree::Union{BitVector, Nothing}`: computed best WPD tree based on the 
    discriminant measure `DM`.
- `DP::Union{AbstractVector{<:AbstractFloat}, Nothing}`: computed discriminant 
    power
- `order::Union{AbstractVector{Integer}, Nothing}`: ordering of `DP` by 
    descending order.
"""
mutable struct LocalDiscriminantBasis
    # to be declared by user
    wt::DiscreteWavelet
    max_dec_level::Union{Integer, Nothing}
    dm::DiscriminantMeasure
    en::EnergyMap
    dp::DiscriminantPower
    top_k::Union{Integer, Nothing}
    n_features::Union{Integer, Nothing}
    # to be computed in fit!
    n::Union{Integer, Nothing}
    Γ::Union{AbstractArray{<:AbstractFloat}, 
             AbstractArray{NamedTuple{(:coef, :weight), Tuple{S1, S2}}} where
                {S1<:Array{T} where T<:AbstractFloat,
                 S2<:Union{AbstractFloat, Array{<:AbstractFloat}}},
             Nothing}
    DM::Union{AbstractArray{<:AbstractFloat}, Nothing}
    cost::Union{AbstractVector{<:AbstractFloat}, Nothing}
    tree::Union{BitVector, Nothing}
    DP::Union{AbstractVector{<:AbstractFloat}, Nothing}
    order::Union{AbstractVector{Integer}, Nothing}
end

"""
    LocalDiscriminantBasis([; 
        wt=wavelet(WT.haar),
        max_dec_level=nothing,
        dm=AsymmetricRelativeEntropy(), em=TimeFrequency(), 
        dp=BasisDiscriminantMeasure(), top_k=nothing,
        n_features=nothing]
    )

Class constructor for `LocalDiscriminantBasis`. 

# Arguments:
- `wt::DiscreteWavelet`: Wavelet used for decomposition of signals. Default is
    set to be `wavelet(WT.haar)`.
- `max_dec_level::Union{Integer, Nothing}`: max level of wavelet packet
    decomposition to be computed. When `max_dec_level=nothing`, the maximum
    transform levels will be used. Default is set to be `nothing`.
- `dm::DiscriminantMeasure`: the discriminant measure for the LDB algorithm. 
    Supported measures are the `AsymmetricRelativeEntropy()`, `LpDistance()`, 
    `SymmetricRelativeEntropy()`, and `HellingerDistance()`. Default is set to
    be `AsymmetricRelativeEntropy()`.
- `en::EnergyMap`: the type of energy map used. Supported maps are 
    `TimeFrequency()` and `ProbabilityDensity()`. Default is set to be 
    `TimeFrequency()`.
- `dp::DiscriminantPower=BasisDiscriminantMeasure()`: the measure of 
    discriminant power among expansion coefficients. Supported measures are 
    `BasisDiscriminantMeasure()`, `FishersClassSeparability()`, and 
    `RobustFishersClassSeparability()`. Default is set to be `BasisDiscriminantMeasure()`.
- `top_k::Union{Integer, Nothing}`: the top-k coefficients used in each node to 
    determine the discriminant measure. When `top_k=nothing`, all coefficients 
    are used to determine the discriminant measure. Default is set to be 
    `nothing`.
- `n_features::Union{Integer, Nothing}`: the dimension of output after 
    undergoing feature selection and transformation. When `n_features=nothing`,
    all features will be returned as output. Default is set to be `nothing`.
"""
function LocalDiscriminantBasis(; wt::DiscreteWavelet=wavelet(WT.haar),
        max_dec_level::Union{Integer, Nothing}=nothing, 
        dm::DiscriminantMeasure=AsymmetricRelativeEntropy(),
        en::EnergyMap=TimeFrequency(), 
        dp::DiscriminantPower=BasisDiscriminantMeasure(), 
        top_k::Union{Integer, Nothing}=nothing, 
        n_features::Union{Integer, Nothing}=nothing)

    return LocalDiscriminantBasis(
        wt, max_dec_level, dm, en, dp, top_k, n_features, 
        nothing, nothing, nothing, nothing, nothing, nothing, nothing
    )
end

"""
    fit!(f, X, y)

Fits the Local Discriminant Basis feature selection algorithm `f` onto the 
signals `X` (or the decomposed signals `Xw`) with labels `y`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit_transform`](@ref),
    [`transform`](@ref), [`inverse_transform`](@ref), [`change_nfeatures`](@ref)
"""
function fit!(f::LocalDiscriminantBasis, X::AbstractArray{S,2}, 
        y::AbstractVector{T}) where {S<:Number, T}

    # basic summary of data
    n, N = size(X)

    # change LocalDiscriminantBasis parameters if necessary
    f.max_dec_level = f.max_dec_level === nothing ? 
        maxtransformlevels(n) : f.max_dec_level
    @assert 1 <= f.max_dec_level <= maxtransformlevels(n)
    
    # wavelet packet decomposition
    Xw = Array{S, 3}(undef, (n,f.max_dec_level+1,N))
    @inbounds begin
        for i in axes(Xw,3)
            Xw[:,:,i] = wpd(X[:,i], f.wt, f.max_dec_level)
        end
    end

    # fit local discriminant basis
    fit!(f, Xw, y)
    return nothing
end

function fit!(f::LocalDiscriminantBasis, Xw::AbstractArray{S,3}, 
        y::AbstractVector{T}) where {S<:Number, T}

    # basic summary of data
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    f.n, L, Nx = size(Xw)

    # change LocalDiscriminantBasis parameters if necessary
    f.top_k = f.top_k === nothing ? f.n : f.top_k
    f.n_features = f.n_features === nothing ? f.n : f.n_features
    f.max_dec_level = f.max_dec_level === nothing ? L-1 : f.max_dec_level

    # parameter checking
    @assert Nx == Ny
    @assert 1 <= f.top_k <= f.n
    @assert 1 <= f.n_features <= f.n
    @assert f.max_dec_level+1 == L
    @assert 1 <= f.max_dec_level <= maxtransformlevels(f.n)
    @assert nc > 1
    @assert isdyadic(f.n)

    # construct energy map for each class
    f.Γ = energy_map(Xw, y, f.en)

    # compute discriminant measure D and obtain tree cost
    f.DM = discriminant_measure(f.Γ, f.dm)
    f.cost = Vector{Float64}(undef, 1<<L-1)
    @inbounds begin
        for i in eachindex(f.cost)
            lθ = floor(Integer, log2(i))    # node level
            θ = i - 1<<lθ                   # node 
            nθ = nodelength(f.n, lθ)        # node length
            rng = (θ*nθ+1):((θ+1)*nθ)
            if f.top_k < nθ
                DMθ = f.DM[rng,lθ+1]        # discriminant measure for node θ
                sort!(DMθ, rev=true)
                f.cost[i] = sum(DMθ[1:f.top_k])
            else
                f.cost[i] = sum(f.DM[rng,lθ+1])
            end
        end
    end

    # select best tree and best set of expansion coefficients
    f.tree = bestbasis_treeselection(f.cost, f.n, :max)
    Xc = bestbasiscoef(Xw, f.tree)

    # obtain and order basis functions by power of discrimination
    (f.DP, f.order) = f.dp == BasisDiscriminantMeasure() ?
        discriminant_power(f.DM, f.tree, f.dp) :
        discriminant_power(Xc, y, f.dp)
    
    return nothing
end

"""
    transform(f, X)

Extract the LDB features on signals `X`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref), 
    [`fit_transform`](@ref), [`inverse_transform`](@ref), [`change_nfeatures`](@ref)
"""
function transform(f::LocalDiscriminantBasis, X::AbstractArray{T,2}) where T
    # check necessary measurements
    n, N = size(X)
    @assert f.max_dec_level !== nothing
    @assert f.top_k !== nothing
    @assert f.n_features !== nothing
    @assert f.n !== nothing
    @assert f.Γ !== nothing
    @assert f.DM !== nothing
    @assert f.cost !== nothing
    @assert f.tree !== nothing
    @assert f.DP !== nothing
    @assert f.order !== nothing
    @assert n == f.n
    @assert 1 <= f.max_dec_level <= maxtransformlevels(f.n)
    @assert 1 <= f.top_k <= f.n
    @assert 1 <= f.n_features <= f.n

    # wpt on X based on given f.tree
    Xc = Array{T, 2}(undef, (n,N))
    @inbounds begin
        for i in axes(Xc,2)
            Xc[:,i] = wpt(X[:,i], f.wt, f.tree)
        end
    end
    return Xc[f.order[1:f.n_features],:]
end

"""
    fit_transform(f, X, y)

Fit and transform the signals `X` with labels `y` based on the LDB class `f`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref),
    [`transform`](@ref), [`inverse_transform`](@ref), [`change_nfeatures`](@ref)
"""
function fit_transform(f::LocalDiscriminantBasis, X::AbstractArray{S,2},
        y::AbstractVector{T}) where {S<:Number, T}

    # get necessary measurements
    n, N = size(X)
    f.max_dec_level = f.max_dec_level===nothing ? 
        maxtransformlevels(n) : f.max_dec_level
    @assert 1 <= f.max_dec_level <= maxtransformlevels(n)

    # wpd on X
    Xw = Array{S, 3}(undef, (n,f.max_dec_level+1,N))
    @inbounds begin
        for i in axes(Xw,3)
            Xw[:,:,i] = wpd(X[:,i], f.wt, f.max_dec_level)
        end
    end

    # fit LDB and return best features
    fit!(f, Xw, y)
    Xc = bestbasiscoef(Xw, f.tree)
    return Xc[f.order[1:f.n_features],:]
end

"""
    inverse_transform(f, x)

Compute the inverse transform on the feature matrix `x` to form the original
signal based on the LDB class `f`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref),
    [`fit_transform`](@ref), [`transform`](@ref), [`change_nfeatures`](@ref)
"""
function inverse_transform(f::LocalDiscriminantBasis, 
        x::AbstractArray{T,2}) where T<:Number

    # get necessary measurements
    @assert size(x,1) == f.n_features
    N = size(x,2)

    # insert the features x into the full matrix X padded with 0's
    Xc = zeros(f.n, N)
    Xc[f.order[1:f.n_features],:] = x

    # iwpt on X
    X = Array{T,2}(undef, (f.n, N))
    @inbounds begin
        for i in axes(Xc, 2)
            X[:,i] = iwpt(Xc[:,i], f.wt, f.tree)
        end
    end
    return X
end

"""
    change_nfeatures(f, x, n_features)

Change the number of features from `f.n_features` to `n_features`. 

Note that if the input `n_features` is larger than `f.n_features`, it results in
the regeneration of signals based on the current `f.n_features` before 
reselecting the features. This will cause additional features to be less 
accurate and effective.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit`](@ref), [`fit_transform`](@ref),
    [`transform`](@ref), [`inverse_transform`](@ref)
"""
function change_nfeatures(f::LocalDiscriminantBasis, x::AbstractArray{T,2},
        n_features::Integer) where T<:Number

    # check measurements
    @assert f.n_features !== nothing
    @assert size(x,1) == f.n_features ||
        throw(ArgumentError("f.n_features and number of rows of x do not match!"))
    @assert 1 <= n_features <= f.n

    # change number of features
    if f.n_features >= n_features
        f.n_features = n_features
        y = x[1:f.n_features,:]
    else
        @warn "Proposed n_features larger than currently saved n_features. Results will be less accurate since inverse_transform and transform is involved."
        X = inverse_transform(f, x)
        f.n_features = n_features
        y = transform(f, X)
    end

    return y
end

end # end module
