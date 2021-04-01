module LocalDiscriminantBasis
export 
    # energy map
    EnergyMap,
    TimeFrequency,
    ProbabilityDensity,
    energy_map,
    # discriminant measures
    DiscriminantMeasure,
    AsymmetricRelativeEntropy,
    SymmetricRelativeEntropy,
    HellingerDistance,
    LpEntropy,
    discriminant_measure,
    # discriminant power
    DiscriminantPower,
    BasisDiscriminantMeasure,
    FishersClassSeparability,
    RobustFishersClassSeparability,
    discriminant_power,
    # local discriminant basis
    ldb

using
    AverageShiftedHistograms,
    Wavelets,
    Parameters,
    Statistics,
    StatsBase

using 
    ..Utils,
    ..WPD,
    ..BestBasis



## ENERGY MAPS
abstract type EnergyMap end
struct TimeFrequency <: EnergyMap end
struct ProbabilityDensity <: EnergyMap end

"""
    energy_map(coef, method)

Returns the Time Frequency Energy map or the Probability Density Energy map
depending on the input `method` (`TimeFrequency()` or `ProbabilityDensity()`).  
"""
function energy_map(coef::AbstractArray{<:Number,3}, method::TimeFrequency)
    n = size(coef, 1)
    L = size(coef, 2) - 1
    @assert isdyadic(n)
    @assert L == maxtransformlevels(n)

    x = coef[:,1,:]
    normalization = sum(mapslices(xᵢ -> sum(xᵢ.^2), x, dims = 1))
    sumsquare_coef = sum(coef.^2, dims = 3)

    return sumsquare_coef / normalization
end

function energy_map(coef::AbstractArray{T,3}, method::ProbabilityDensity) where 
        T<:Number

    n = size(coef, 1)
    L = size(coef, 2) - 1
    N = size(coef, 3)
    @assert isdyadic(n)
    @assert L == maxtransformlevels(n)

    Γ = Array{T,2}(undef, (n,L+1))
    for j in axes(coef, 2)
        γᵢ = Vector{T}(undef, n)
        for i in eachindex(γᵢ)
            x = coef[i,j,:]                      # segment of coef
            nbins = ceil(Integer, (30*N)^(1/5))  # number of bins per histogram
            # number of histograms using M/nbins, where M=100 is arbitrary
            mbins = ceil(Integer, 100/nbins)     

            σ = std(x)                           # standard deviatio of x
            s = 0.5                              
            δ = (maximum(x) - minimum(x) + σ)/((nbins+1) * mbins - 1)               
            rng = (minimum(x) - s*σ):δ:(maximum(x) + s*σ)   
            # compute EPDF using ASH                        
            epdf = ash(x, rng = rng, m = mbins, kernel = Kernels.triangular)   

            # calculate expectation E[Zᵢ²|Y=y] = ∫ z² q̂ᵢ(z) dz ≈ ∑ zₖ² q̂ᵢ(zₖ) δ
            E = 0                                                                   
            for k in 1:N
                E += (x[k]^2) * AverageShiftedHistograms.pdf(epdf, x[k]) * δ
            end
            γᵢ[i] = E
        end
        Γ[:,j] = γᵢ/sum(γᵢ)
    end
    return Γ
end

## DISCRIMINANT MEASURES
abstract type DiscriminantMeasure end
struct AsymmetricRelativeEntropy <: DiscriminantMeasure end
struct SymmetricRelativeEntropy <: DiscriminantMeasure end
struct HellingerDistance <: DiscriminantMeasure end
@with_kw struct LpEntropy <: DiscriminantMeasure 
    p::Number = 2
end

"""
    discriminant_measure(Γ, dm)

Returns the discriminant measure calculated from the energy maps.
"""
function discriminant_measure(Γ::AbstractArray{T,3}, 
        dm::DiscriminantMeasure) where T<:Number

    (n, levels, C) = size(Γ)
    @assert C > 1       # ensure more than 1 class

    D = zeros(T, (n,levels))
    for i in 1:(C-1)
        for j in (i+1):C
            D += discriminant_measure(Γ[:,:,i], Γ[:,:,j], dm)
        end
    end

    return D
end

function discriminant_measure(Γ₁::AbstractArray{T,2}, Γ₂::AbstractArray{T,2}, 
        dm::DiscriminantMeasure) where T<:Number

    @assert size(Γ₁) == size(Γ₂)

    D = Array{T, 2}(undef, size(Γ₁))
    @inbounds begin
        for i in eachindex(Γ₁, Γ₂)
            D[i] = discriminant_measure(Γ₁[i], Γ₂[i], dm)
        end
    end

    return D
end

# Asymmetric Relative Entropy
function discriminant_measure(p::T, q::T, dm::AsymmetricRelativeEntropy) where 
        T<:Number

    @assert p > 0 && q > 0
    return p * log(p/q)
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

# Lᵖ Entropy
function discriminant_measure(p::T, q::T, dm::LpEntropy) where T<:Number
    return (p - q)^dm.p
end

## DISCRIMINATION POWER
abstract type DiscriminantPower end
struct BasisDiscriminantMeasure <: DiscriminantPower end
struct FishersClassSeparability <: DiscriminantPower end
struct RobustFishersClassSeparability <: DiscriminantPower end

"""
    discriminant_power(coefs, tree, dp)

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
        tree::BitVector, dp::FishersClassSeparability) where {T<:Number, S}

    n = size(coefs,1)                             # signal length
    @assert length(tree) == size(coefs,1) - 1     # ensure tree is of right size

    classes = unique(y)
    C = length(coefs)                             # number of classes
    
    Nᵢ = Array{T,1}(undef, C)
    Eαᵢ = Array{T,2}(undef, (n,C))                # mean of each entry
    Varαᵢ = Array{T,2}(undef, (n,C))              # variance of each entry
    for (i, c) in enumerate(classes)
        idx = findall(yᵢ -> yᵢ == c, y)
        Nᵢ[i] = length(idx)
        Eαᵢ[:,i] = mean(coefs[:, idx], dims = 2)
        Varαᵢ[:,i] = var(coefs[:, idx], dims = 2)
    end
    Eα = mean(Eαᵢ, dims = 2)                      # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                             # proportions of each class

    power = ((Eαᵢ - (Eα .* Eαᵢ)).^2 * pᵢ) ./ (Varαᵢ * pᵢ)
    order = sortperm(power, rev = true)

    return (power, order)
end

function discriminant_power(coefs::AbstractArray{T,2}, y::AbstractVector{S},
        tree::BitVector, dp::RobustFishersClassSeparability) where {T<:Number,S}

    n = size(coefs,1)                            # signal length
    @assert length(tree) == size(coefs,1) - 1    # ensure tree is of right size

    classes = unique(y)
    C = length(classes)

    Nᵢ = Array{T,1}(undef, C)
    Medαᵢ = Array{T,2}(undef, (n,C))             # mean of each entry
    Madαᵢ = Array{T,2}(undef, (n,C))             # variance of each entry
    for (i, c) in enumerate(classes)
        idx = findall(yᵢ -> yᵢ == c, y)
        Nᵢ[i] = length(idx)
        Medαᵢ[:,i] = median(coefs[:, idx], dims = 2)
        Madαᵢ[:,i] = mapslices(x -> mad(x, normalize = false), coefs[:, idx], 
            dims = 2)
    end
    Medα = median(Medαᵢ, dims = 2)               # overall mean of each entry
    pᵢ = Nᵢ / sum(Nᵢ)                            # proportions of each class

    power = ((Medαᵢ - (Medα .* Medαᵢ)).^2 * pᵢ) ./ (Madαᵢ * pᵢ)
    order = sortperm(power, rev = true)

    return (power, order)
end

## LOCAL DISCRIMINANT BASIS
"""
    ldb(X, y, wt[; dm=AsymmetricRelativeEntropy(), energy=TimeFrequency(),
        dp=BasisDiscriminantMeasure(), topk=size(X,1), m=size(X,1)])
"""
function ldb(X::AbstractArray{S,2}, y::AbstractVector{T}, wt::DiscreteWavelet; 
    dm::DiscriminantMeasure=AsymmetricRelativeEntropy(), 
    energy::EnergyMap=TimeFrequency(), 
    dp::DiscriminantPower=BasisDiscriminantMeasure(), topk::Integer=size(X,1), 
    m::Integer=size(X,1)) where {S<:Number, T}
    
    classes = unique(y)
    C = length(classes)
    n, N = size(X)

    @assert size(X,2) == length(y) 
    @assert 1 <= topk <= n  
    @assert 1 <= m <= n       
    @assert C > 1                  # checking number of classes > 1
    @assert isdyadic(n)            # checking if input signals of length 2ᴸ
    
    # compute wpt for each signal and construct energy map for each class
    L = maxtransformlevels(n) + 1
    X_wpt = Array{Float64, 3}(undef, (n, L, N))
    ỹ = similar(y)
    Γ = Array{Float64, 3}(undef, (n, L, C))
    j = 1
    for (i,c) in enumerate(classes)
        # settle indexing
        idx = findall(yᵢ -> yᵢ == c, y)
        Nc = length(idx)
        rng = j:(j+Nc-1)
        j += Nc
        # wavelet packet decomposition and energy map
        X_wpt[:,:,rng] = wpd(X[:,idx], wt)
        ỹ[rng] .= c
        Γ[:,:,i] = energy_map(X_wpt[:,:,rng], energy)
    end

    # compute discriminant measure D and obtain Δ
    D = discriminant_measure(Γ, dm)
    Δ = Vector{Float64}(undef, 2^L-1)
    for i in eachindex(Δ)
        level = floor(Integer, log2(i))
        node = i - 2^level 
        len = nodelength(n, level)
        rng = (node*len+1):((node+1)*len)
        if topk < len               # sum up top k coefficients
            node_dm = D[rng, level+1]
            sort!(node_dm, rev = true)
            Δ[i] = sum(node_dm[1:topk])
        else                        # sum of all coefficients
            Δ[i] = sum(D[rng, level+1])      
        end
    end

    # select best tree and best set of expansion coefficients
    besttree = ldb_besttree(Δ, n)
    coefs = bestbasiscoef(X_wpt, besttree)

    # obtain and order basis functions by power of discrimination
    (power, order) = dp == BasisDiscriminantMeasure() ? 
        discriminant_power(D, besttree, dp) : 
        discriminant_power(coefs, ỹ, besttree, dp)

    return (coefs[order[1:m],:], ỹ, besttree, power[order[1:m]], order[1:m])        
end

# select best ldb tree
function ldb_besttree(Δ::AbstractVector{T}, n::Integer) where T<:Number
    @assert length(Δ) == 2*n - 1
    bt = trues(n-1)
    for i in reverse(eachindex(bt))
        δ = Δ[i<<1] + Δ[(i<<1)+1]
        if δ > Δ[i]     # child cost > parent cost
            Δ[i] = δ
        else
            BestBasis.delete_subtree!(bt, i)                                                    
        end
    end
    return bt
end


end # end module
