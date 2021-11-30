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
function discriminant_measure(Γ::AbstractArray{NamedTuple{(:coef, :weight), Tuple{S1,S2}},1},
                              dm::SignaturesDM) where 
                             {S1<:Array{<:Number}, 
                              S2<:Union{AbstractFloat,Array{<:AbstractFloat}}}
    # Basic summary of data
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

    power = getbasiscoef(D, tree)
    order = sortperm(power, rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T,2}, 
                            y::AbstractVector{S}, 
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

function discriminant_power(coefs::AbstractArray{T,2}, 
                            y::AbstractVector{S},
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
