import Combinatorics: combinations

## ---------- DISCRIMINANT MEASURES ----------
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

Asymmetric Relative Entropy discriminant measure for the Probability Density and Time
Frequency based energy maps. This measure is also known as cross entropy and
Kullback-Leibler divergence.

Equation: 

``D(p,q) = \sum p(x) \log \frac{p(x)}{q(x)}`` 

where ``\int_{-\infty}^{\infty} p(t) dt = \int_{-\infty}^{\infty} q(t) dt = 1`` (``p(t)``
and ``q(t)`` are probability density functions.)
"""
struct AsymmetricRelativeEntropy <: ProbabilityDensityDM end

@doc raw"""
    SymmetricRelativeEntropy <: ProbabilityDensityDM

Symmetric Relative Entropy discriminant measure for the Probability Density and Time
Frequency energy maps. Similar idea to the Asymmetric Relative Entropy, but this aims to
make the measure more symmetric.

Equation: Denote the Asymmetric Relative Entropy as ``D_A(p,q)``, then

``D(p,q) = D_A(p,q) + D_A(q,p) = \sum p(x) \log \frac{p(x)}{q(x)} + q(x) \log
\frac{q(x)}{p(x)}``

where ``\int_{-\infty}^{\infty} p(t) dt = \int_{-\infty}^{\infty} q(t) dt = 1`` (``p(t)``
and ``q(t)`` are probability density functions.)

**See also:** [`AsymmetricRelativeEntropy`](@ref)
"""
struct SymmetricRelativeEntropy <: ProbabilityDensityDM end

@doc raw"""
    LpDistance <: ProbabilityDensityDM

``\ell^p`` Distance discriminant measure for the Probability Density and Time 
Frequency based energy maps. The default ``p`` value is set to 2.

Equation: 

``W(q,r) = ||q-r||_p^p = \sum_{i=1}^n (q_i - r_i)^p``
"""
@with_kw struct LpDistance{T<:Real} <: ProbabilityDensityDM 
    p::T = 2
end

@doc raw"""
    HellingerDistance <: ProbabilityDensityDM

Hellinger Distance discriminant measure for the Probability Density energy 
map.

Equation: 

``H(p,q) = \sum_{i=1}^n (\sqrt{p_i} - \sqrt{q_i})^2``
"""
struct HellingerDistance <: ProbabilityDensityDM end

@doc raw"""
    EarthMoverDistance <: SignaturesDM

Earth Mover Distance discriminant measure for the Signatures energy map.

Equation: 

``E(P,Q) = \frac{\sum_{k=1}^{m+n+1} |\hat p_k - \hat q_k| (r_{k+1} - r_k)}{w_\Sigma}``

where 
- ``r_1, r_2, \ldots, r_{m+n}`` is the sorted list of ``p_1, \ldots, p_m, q_1, \ldots, q_n``
- ``\hat p_k = \sum_{p_i \leq r_k} w_{p_i}``
- ``\hat q_k = \sum_{q_i \leq r_k} w_{q_i}``
"""
struct EarthMoverDistance <: SignaturesDM end

"""
    discriminant_measure(Γ, dm)

Computes the discriminant measure of each subspace calculated from the energy maps.

# Arguments
- `Γ`: Energy map computed from `energy_map` function. The data structures of `Γ`, depending
  on their corresponding energy map, should be:
    - `TimeFrequency()`: `AbstractArray{T,3}` for 1D signals or `AbstractArray{T,4}` for 2D
      signals.
    - `ProbabilityDensity()`: `AbstractArray{T,4}` for 1D signals or `AbstractArray{T,5}`
      for 2D signals.
    - `Signatures()`: `AbstractVector{NamedTuple{(:coef, :weight), Tuple{S₁,S₂}}}`.
- `dm::DiscriminantMeasure`: Type of Discriminant Measure. The type of `dm` must match the
  type of `Γ`.

# Returns
- `D::Array{T}`: Discriminant measure at each coefficient of the decomposed signals.

# Examples
```julia
using Wavelets, WaveletsExt

X, y = generateclassdata(ClassData(:tri, 5, 5, 5))
Xw = wpdall(X, wavelet(WT.haar))

Γ = energy_map(Xw, y, TimeFrequency()); discriminant_measure(Γ, AsymmetricRelativeEntropy())
Γ = energy_map(Xw, y, ProbabilityDensity()); discriminant_measure(Γ, LpDistance())
Γ = energy_map(Xw, y, Signatures()); discriminant_measure(Γ, EarthMoverDistance())
```

**See also:** [`pairwise_discriminant_measure`](@ref)
"""
function discriminant_measure(Γ::AbstractArray{T}, dm::ProbabilityDensityDM) where 
                              T<:AbstractFloat
    # Basic summary of data
    N = ndims(Γ)
    nc = size(Γ,N)
    @assert 3 ≤ N ≤ 5
    @assert nc > 1
    #=======================================================================================
    Classifying the type of signals and energy maps based on Γ

    1. ndims(Γ)=3                           : Time frequency energy map on 1D signals
    2. ndims(Γ)=4 and size(Γ,3) small (<100): Time frequency energy map on 2D signals
    3. ndims(Γ)=4 and size(Γ,3) large (≥100): Probability density energy map on 1D signals
    4. ndims(Γ)=5                           : Probability density energy map on 2D signals
    =======================================================================================#
    if (N==3) || (N==4 && size(Γ,3)≥100)
        sz = [size(Γ,1)]
        L = size(Γ,2)
        pdf_len = N==3 ? 1 : size(Γ,3)
    elseif (N==4 && size(Γ,3)<100) || (N==5)
        sz = size(Γ)[1:2]
        L = size(Γ,3)
        pdf_len = N==5 ? size(Γ,4) : 1
    else
        throw(ArgumentError("Γ has uninterpretable dimensions/unknown size."))
    end

    # --- Compute pairwise discriminant measure ---
    D = zeros(T, (sz...,L))
    for (i,j) in combinations(1:nc,2)
        if N==3
            Γᵢ = @view Γ[:,:,i]         # Energy map for i-th class
            Γⱼ = @view Γ[:,:,j]         # Energy map for j-th class
        elseif N==4
            Γᵢ = @view Γ[:,:,:,i]       # Energy map for i-th class
            Γⱼ = @view Γ[:,:,:,j]       # Energy map for j-th class
        else # N==5
            Γᵢ = @view Γ[:,:,:,:,i]     # Energy map for i-th class
            Γⱼ = @view Γ[:,:,:,:,j]     # Energy map for j-th class
        end
        D += pairwise_discriminant_measure(Γᵢ, Γⱼ, dm)
    end
    return D
end

# discriminant measure for EMD
function discriminant_measure(Γ::AbstractVector{NamedTuple{(:coef, :weight), Tuple{S₁,S₂}}},
                              dm::SignaturesDM) where 
                             {S₁<:Array{T} where T<:AbstractFloat, 
                              S₂<:Union{T,Array{T}} where T<:AbstractFloat}
    # Basic summary of data
    nc = length(Γ)
    @assert nc > 1
    T = eltype(Γ[1][:coef])
    sz = size(Γ[1][:coef])[1:end-2]
    L = size(Γ[1][:coef])[end-1]

    D = zeros(T, (sz...,L))
    for (Γᵢ,Γⱼ) in combinations(Γ,2)
        D += pairwise_discriminant_measure(Γᵢ, Γⱼ, dm)
    end
    return D
end

# discriminant measure between 2 energy maps
"""
    pairwise_discriminant_measure(Γ₁, Γ₂, dm)

Computes the discriminant measure between 2 classes based on their energy maps.

# Arguments
- `Γ₁::AbstractArray{T} where T<:AbstractFloat`: Energy map for class 1.
- `Γ₂::AbstractArray{T} where T<:AbstractFloat`: Energy map for class 2.
- `dm::DiscriminantMeasure`: Type of discriminant measure.

# Returns
- `::Array{T}`: Discriminant measure between `Γ₁` and `Γ₂`.
"""
function pairwise_discriminant_measure(Γ₁::AbstractArray{T}, Γ₂::AbstractArray{T}, 
                                       dm::ProbabilityDensityDM) where T<:AbstractFloat
    # parameter checking and basic summary
    N = ndims(Γ₁)
    @assert 2 ≤ ndims(Γ₁) ≤ 4
    @assert size(Γ₁) == size(Γ₂)
    #=======================================================================================
    Classifying the type of signals and energy maps based on Γ

    1. ndims(Γ)=2                           : Time frequency energy map on 1D signals
    2. ndims(Γ)=3 and size(Γ,3) small (<100): Time frequency energy map on 2D signals
    3. ndims(Γ)=3 and size(Γ,3) large (≥100): Probability density energy map on 1D signals
    4. ndims(Γ)=4                           : Probability density energy map on 2D signals
    =======================================================================================#
    if (N==2) || (N==3 && size(Γ₁,3)≥100)
        sz = [size(Γ₁,1)]
        L = size(Γ₁,2)
    elseif (N==3 && size(Γ₁,3)<100) || (N==4)
        sz = size(Γ₁)[1:2]
        L = size(Γ₁,3)
    else
        throw(ArgumentError("Γ has uninterpretable dimensions/unknown size."))
    end

    # --- Pairwise discriminant measure for each element ---
    D = zeros(T, (sz...,L))
    slice_size = prod([sz...,L])        # Number of elements in each slice of the pdf
    map_size = prod(size(Γ₁))           # Number of elements in entire energy map
    for i in 1:slice_size
        for j in i:slice_size:map_size
            D[i] += pairwise_discriminant_measure(Γ₁[j], Γ₂[j], dm)
        end
    end
    return D
end

# discriminant measure between 2 nergy maps for EMD
function pairwise_discriminant_measure(Γ₁::NamedTuple{(:coef, :weight), Tuple{S₁,S₂}}, 
                                       Γ₂::NamedTuple{(:coef, :weight), Tuple{S₁,S₂}}, 
                                       dm::SignaturesDM) where
                                      {S₁<:Array{T} where T<:AbstractFloat, 
                                       S₂<:Union{T,Array{T}} where T<:AbstractFloat}

    # parameter checking and basic summary
    N = ndims(Γ₁.coef)
    T = eltype(Γ₁.coef)
    sz = size(Γ₁.coef)[1:end-2]
    L = size(Γ₁.coef)[end-1]
    @assert isa(Γ₁.weight, AbstractFloat) || size(Γ₁.coef) == size(Γ₁.weight)
    @assert isa(Γ₂.weight, AbstractFloat) || size(Γ₂.coef) == size(Γ₂.weight)
    @assert typeof(Γ₁.weight) == typeof(Γ₂.weight)
    @assert sz == size(Γ₂.coef)[1:end-2]
    @assert L == size(Γ₂.coef)[end-1]

    D = Array{T,N-1}(undef, (sz...,L))
    slice_size = prod([sz...,L])                # Number of elements for each signal's coefficients
    for i in 1:slice_size
        # Signatures
        if isa(Γ₁.weight, AbstractFloat)        # Equal weight
            P = (coef = Γ₁.coef[i:slice_size:end], weight = Γ₁.weight)
            Q = (coef = Γ₂.coef[i:slice_size:end], weight = Γ₂.weight)
        else                                    # Probability density weight
            P = (coef = Γ₁.coef[i:slice_size:end], weight = Γ₁.weight[i:slice_size:end])
            Q = (coef = Γ₂.coef[i:slice_size:end], weight = Γ₂.weight[i:slice_size:end])
        end
        D[i] = pairwise_discriminant_measure(P, Q, dm)
    end
    return D
end

# Asymmetric Relative Entropy
"""
    pairwise_discriminant_measure(p, q, dm)

Computes the discriminant measure between 2 classes at an index ``i``, where `Γ₁[i] = p` and
`Γ₂[i] = q`.

# Arguments
- `p::T where T<:AbstractFloat`: Coefficient at index ``i`` from `Γ₁`.
- `q::T where T<:AbstractFloat`: Coefficient at index ``i`` from `Γ₂`.
- `dm::DiscriminantMeasure`: Type of discriminant measure.

# Returns
- `::T`: Discriminant measure between `p` and `q`.
"""
function pairwise_discriminant_measure(p::T, q::T, dm::AsymmetricRelativeEntropy) where 
                                       T<:AbstractFloat
    @assert p ≥ 0 && q ≥ 0
    return (p==0 || q==0) ? 0 : p * log(p/q)
end

# Symmetric Relative Entropy
function pairwise_discriminant_measure(p::T, q::T, dm::SymmetricRelativeEntropy) where 
                                       T<:AbstractFloat
    return pairwise_discriminant_measure(p, q, AsymmetricRelativeEntropy()) + 
           pairwise_discriminant_measure(q, p, AsymmetricRelativeEntropy())
end

# Hellinger Distance
function pairwise_discriminant_measure(p::T, q::T, dm::HellingerDistance) where 
                                       T<:AbstractFloat
    return (sqrt(p) - sqrt(q))^2
end

# Lᵖ Distance
function pairwise_discriminant_measure(p::T, q::T, dm::LpDistance) where T<:AbstractFloat
    return (p - q)^dm.p
end

# Earth Mover Distance
function pairwise_discriminant_measure(P::NamedTuple{(:coef, :weight), Tuple{S1,S2}}, 
                                       Q::NamedTuple{(:coef, :weight), Tuple{S1, S2}}, 
                                       dm::EarthMoverDistance) where
                                      {S1<:Vector{T} where T<:AbstractFloat, 
                                       S2<:Union{T,Vector{T}} where T<:AbstractFloat}
    # assigning tuple signatures into coef and weight
    p, w_p = P
    q, w_q = Q

    # sort signature values
    p_order = sortperm(p)
    q_order = sortperm(q)
    p = p[p_order]
    q = q[q_order]
    w_p = isa(w_p, AbstractFloat) ? repeat([w_p], length(p)) : w_p[p_order]
    w_q = isa(w_q, AbstractFloat) ? repeat([w_q], length(q)) : w_q[q_order]

    # concatenate p and q, then sort them
    r = [p; q]
    sort!(r)

    # compute emd
    n = length(r)
    emd = 0
    for i in 1:(n-1)
        # get total weight of p and q less than or equal to r[i]
        ∑w_p = sum(w_p[p .≤ r[i]])
        ∑w_q = sum(w_q[q .≤ r[i]])
        # add into emd
        @inbounds emd += abs(∑w_p - ∑w_q) * (r[i+1] - r[i])
    end
    emd /= (sum(w_p) + sum(w_q))
    return emd
end

## ---------- DISCRIMINATION POWER ----------
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

This is the discriminant measure of a single basis function computed in a previous step to
construct the energy maps.
"""
struct BasisDiscriminantMeasure <: DiscriminantPower end

@doc raw"""
    FishersClassSeparability <: DiscriminantPower

The Fisher's class separability of the expansion coefficients in the basis function.

Equation: ``\frac{\sum_{c=1}^C \pi_c\{{\rm mean}_i(\alpha_{\lambda,i}^{(c)}) - {\rm mean}_c
\cdot {\rm mean}_i(\alpha_{\lambda,i}^{(c)})\}^2}{\sum_{c=1}^C \pi_c {\rm
var}_i(\alpha_{\lambda,i}^{(c)})}``
"""
struct FishersClassSeparability <: DiscriminantPower end

@doc raw"""
    RobustFishersClassSeparability <: DiscriminantPower

The robust version of Fisher's class separability of the expansion coefficients in the basis
function.

Equation: ``\frac{\sum_{c=1}^C \pi_c\{{\rm med}_i(\alpha_{\lambda,i}^{(c)}) - {\rm med}_c
\cdot {\rm med}_i(\alpha_{\lambda,i}^{(c)})\}^2}{\sum_{c=1}^C \pi_c {\rm
mad}_i(\alpha_{\lambda,i}^{(c)})}``
"""
struct RobustFishersClassSeparability <: DiscriminantPower end

"""
    discriminant_power(D, tree, dp)
    discriminant_power(coefs, y, dp)

Returns the discriminant power of each leaf from the local discriminant basis (LDB) tree. 

# Arguments
- `D::AbstractArray{T} where T<:AbstractFloat`: Discriminant measures.
- `tree::BitVector`: Best basis tree for selecting coefficients with largest discriminant
  measures.
- `coefs::AbstractArray{T} where T<:AbstractFloat`: Best basis coefficients for the input
  signals.
- `y::AbstractVector{S} where S`: Labels corresponding to each signal in `coefs`.
- `dp::DiscriminantPower`: The measure for discriminant power. 

!!! note
    `discriminant_power(D, tree, dp)` only works for `dp = BasisDiscriminantMeasure()`,
    whereas `discriminant_power(coefs, y, dp)` works for `dp = FishersClassSeparability()`
    and `dp = RobustFishersClassSeparability()`.

# Returns
- `power::Array{T}`: The discriminant power at each index of `D` or `coefs`.
- `order::Vector{T}`: The order of discriminant power in descending order.
"""
function discriminant_power(D::AbstractArray{T}, tree::BitVector, 
                            dp::BasisDiscriminantMeasure) where T<:AbstractFloat
    @assert 2 ≤ ndims(D) ≤ 3
    N = ndims(D)
    sz = size(D)[1:end-1]
    tmp = Array{T,N-1}(undef, sz)
    @assert isvalidtree(tmp, tree)

    power = getbasiscoef(D, tree)
    order = sortperm(vec(power), rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T}, y::AbstractVector{S}, 
                            dp::FishersClassSeparability) where {T<:AbstractFloat, S}
    N = ndims(coefs)
    @assert 2 ≤ N ≤ 3
    sz = size(coefs)[1:end-1]           # signal length
    
    C = unique(y)
    Nc = length(C)                      # number of classes
    
    Nᵢ = Array{T,1}(undef, Nc)
    Eαᵢ = Array{T,N}(undef, (sz...,Nc))                # mean of each entry
    Varαᵢ = Array{T,N}(undef, (sz...,Nc))              # variance of each entry
    for (i, c) in enumerate(C)
        idx = findall(yᵢ -> yᵢ==c, y)
        Nᵢ[i] = length(idx)
        if N==2
            coefsᵢ = coefs[:,idx]
            Eαᵢ[:,i] = mean(coefsᵢ, dims = N)
            Varαᵢ[:,i] = var(coefsᵢ, dims = N)
        elseif N==3
            coefsᵢ = coefs[:,:,idx]
            Eαᵢ[:,:,i] = mean(coefsᵢ, dims = N)
            Varαᵢ[:,:,i] = var(coefsᵢ, dims = N)
        end
    end
    Eα = mean(Eαᵢ, dims = N)                      # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                             # proportions of each class

    if N==2     # For 1D signals, can be done via matrix multiplication
        power = ((Eαᵢ - (Eα .* Eαᵢ)).^2 * pᵢ) ./ (Varαᵢ * pᵢ)
    else        # For 2D signals, requires some manual work
        pᵢ = reshape(pᵢ,1,1,:)
        power = sum((Eαᵢ-(Eα.*Eαᵢ)).^2 .* pᵢ, dims=N) ./ sum(Varαᵢ.*pᵢ, dims=N)
        power = reshape(power, sz...)
    end
    order = sortperm(vec(power), rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T}, y::AbstractVector{S},
                            dp::RobustFishersClassSeparability) where {T<:AbstractFloat,S}
    N = ndims(coefs)
    @assert 2 ≤ N ≤ 3
    sz = size(coefs)[1:end-1]           # signal length
    
    C = unique(y)
    Nc = length(C)                      # number of classes

    Nᵢ = Array{T,1}(undef, Nc)
    Medαᵢ = Array{T,N}(undef, (sz...,Nc))              # mean of each entry
    Madαᵢ = Array{T,N}(undef, (sz...,Nc))              # variance of each entry
    for (i, c) in enumerate(C)
        idx = findall(yᵢ -> yᵢ==c, y)
        Nᵢ[i] = length(idx)
        if N==2
            coefsᵢ = coefs[:,idx]
            Medαᵢ[:,i] = median(coefsᵢ, dims = N)
            Madαᵢ[:,i] = mapslices(x -> mad(x, normalize=false), coefsᵢ, dims = N)
        elseif N==3
            coefsᵢ = coefs[:,:,idx]
            Medαᵢ[:,:,i] = median(coefsᵢ, dims = N)
            Madαᵢ[:,:,i] = mapslices(x -> mad(x, normalize=false), coefsᵢ, dims = N)
        end
    end
    Medα = median(Medαᵢ, dims = N)               # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                            # proportions of each class

    if N==2     # For 1D signals, can be done via matrix multiplication
        power = ((Medαᵢ - (Medα.*Medαᵢ)).^2 *pᵢ) ./ (Madαᵢ * pᵢ)
    else        # For 2D signals, requires some manual work
        pᵢ = reshape(pᵢ,1,1,:)
        power = sum((Medαᵢ-(Medα.*Medαᵢ)).^2 .* pᵢ, dims=N) ./ sum(Madαᵢ.*pᵢ, dims=N)
        power = reshape(power, sz...)
    end
    order = sortperm(vec(power), rev = true)

    return (power, order)
end
