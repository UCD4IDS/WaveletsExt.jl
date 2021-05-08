module LDB
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
    LocalDiscriminantBasis,
    fit!,
    fit_transform,
    transform,
    inverse_transform,
    change_nfeatures

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
    LocalDiscriminantBasis

Class type for the Local Discriminant Basis (LDB), a feature selection algorithm
developed by N. Saito and R. Coifman in "Local Discriminant Bases and Their
Applications" in the Journal of Mathematical Imaging and Vision, Vol 5, 337-358
(1995). This function takes in the input signals and their 
respective 

# Parameters and Attributes:
- `wt::DiscreteWavelet`: a discrete wavelet for transform purposes
- `dm::DiscriminantMeasure`: the discriminant measure for the LDB algorithm. 
    Supported measures are the `AsymmetricRelativeEntropy()`, `LpEntropy()`,
    `SymmetricRelativeEntropy()`, and `HellingerDistance()`
- `en::EnergyMap`: the type of energy map used. Supported maps are 
    `TimeFrequency()` and `ProbabilityDensity()`.
- `dp::DiscriminantPower()`: the measure of discriminant power among expansion
    coefficients. Supported measures are `BasisDiscriminantMeasure()`,
    `FishersClassSeparability()`, and `RobustFishersClassSeparability()`. 
- `top_k::Union{Integer, Nothing}`: the top-k coefficients used in each node to 
    determine the discriminant measure.
- `n_features::Union{Integer, Nothing}`: the dimension of output after 
    undergoing feature selection and transformation.
- `n::Union{Integer, Nothing}`: length of signal
- `Γ::Union{AbstractArray{<:AbstractFloat}, Nothing}`: computed energy map
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
    dm::DiscriminantMeasure
    en::EnergyMap
    dp::DiscriminantPower
    top_k::Union{Integer, Nothing}
    n_features::Union{Integer, Nothing}
    # to be computed in fit!
    n::Union{Integer, Nothing}
    Γ::Union{AbstractArray{<:AbstractFloat}, Nothing}
    DM::Union{AbstractArray{<:AbstractFloat}, Nothing}
    cost::Union{AbstractVector{<:AbstractFloat}, Nothing}
    tree::Union{BitVector, Nothing}
    DP::Union{AbstractVector{<:AbstractFloat}, Nothing}
    order::Union{AbstractVector{Integer}, Nothing}
end

"""
    LocalDiscriminantBasis(wt[; dm=AsymmetricRelativeEntropy(),
        em=TimeFrequency(), dp=BasisDiscriminantMeasure(), top_k=nothing,
        n_features=nothing])

Class constructor for `LocalDiscriminantBasis`. 

# Arguments:
- `wt::DiscreteWavelet`: Wavelet used for decomposition of signals.
- `dm::DiscriminantMeasure`: the discriminant measure for the LDB algorithm. 
    Supported measures are the `AsymmetricRelativeEntropy()`, `LpEntropy()`, 
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
function LocalDiscriminantBasis(wt::DiscreteWavelet; 
        dm::DiscriminantMeasure=AsymmetricRelativeEntropy(),
        en::EnergyMap=TimeFrequency(), 
        dp::DiscriminantPower=BasisDiscriminantMeasure(), 
        top_k::Union{Integer, Nothing}=nothing, 
        n_features::Union{Integer, Nothing}=nothing)

    return LocalDiscriminantBasis(
        wt, dm, en, dp, top_k, n_features, 
        nothing, nothing, nothing, nothing, nothing, nothing, nothing
    )
end

"""
    fit!(f, X, y)

Fits the Local Discriminant Basis feature selection algorithm `f` onto the 
signals `X` (or the decomposed signals `Xw`) with labels `y`.
"""
function fit!(f::LocalDiscriminantBasis, X::AbstractArray{S,2}, 
        y::AbstractVector{T}) where {S<:Number, T}

    Xw = cat([wpd(X[:,i], f.wt) for i in axes(X,2)]..., dims=3)
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

    # parameter checking
    @assert Nx == Ny
    @assert 1 <= f.top_k <= f.n
    @assert 1 <= f.n_features <= f.n
    @assert nc > 1
    @assert isdyadic(f.n)

    # construct energy map for each class
    ỹ = similar(y)
    f.Γ = Array{Float64, 3}(undef, (f.n, L, nc))
    for (i,cᵢ) in enumerate(c)
        idx = findall(yᵢ -> yᵢ==cᵢ, y)
        f.Γ[:,:,i] = energy_map(Xw[:,:,idx], f.en)
    end

    # compute discriminant measure D and obtain 
    f.DM = discriminant_measure(f.Γ, f.dm)
    f.cost = Vector{Float64}(undef, 1<<L-1)
    for i in eachindex(f.cost)
        lθ = floor(Integer, log2(i))    # node level
        θ = i - 1<<lθ                   # node 
        nθ = nodelength(f.n, lθ)          # node length
        rng = (θ*nθ+1):((θ+1)*nθ)
        if f.top_k < nθ
            DMθ = f.DM[rng,lθ+1]
            sort!(DMθ, rev=true)
            f.cost[i] = sum(DMθ[1:f.top_k])
        else
            f.cost[i] = sum(f.DM[rng,lθ+1])
        end
    end

    # select best tree and best set of expansion coefficients
    f.tree = bestbasis_treeselection(f.cost, f.n, :max)
    coefs = bestbasiscoef(Xw, f.tree)

    # obtain and order basis functions by power of discrimination
    (f.DP, f.order) = f.dp == BasisDiscriminantMeasure() ?
        discriminant_power(f.DM, f.tree, f.dp) :
        discriminant_power(coefs, y, f.tree, f.dp)
    
    return nothing
end

"""
    transform(f, X)

Extract the LDB features on signals `X`.
"""
function transform(f::LocalDiscriminantBasis, X::AbstractArray{<:Number,2})
    Xw = cat([wpd(X[:,i], f.wt) for i in axes(X,2)]..., dims=3)
    coefs = bestbasiscoef(Xw, f.tree)
    return coefs[f.order[1:f.n_features],:]
end

"""
    fit_transform(f, X, y)

Fit and transform the signals `X` with labels `y` based on the LDB class `f`.
"""
function fit_transform(f::LocalDiscriminantBasis, X::AbstractArray{S,2},
    y::AbstractVector{T}) where {S<:Number, T}

    Xw = cat([wpd(X[:,i], f.wt) for i in axes(X,2)]..., dims=3)
    fit!(f, Xw, y)
    coefs = bestbasiscoef(Xw, f.tree)
    return coefs[f.order[1:f.n_features],:], y
end

"""
    inverse_transform(f, x)

Compute the inverse transform on the feature matrix `x` to form the original
signal based on the LDB class `f`.
"""
function inverse_transform(f::LocalDiscriminantBasis, 
        x::AbstractArray{T,2}) where T<:Number

    @assert size(x,1) == f.n_features
    Nx = size(x,2)
    Xc = zeros(f.n, Nx)
    Xc[f.order[1:f.n_features],:] = x
    X = hcat([iwpt(Xc[:,i], f.wt, f.tree) for i in axes(Xc,2)]...)
    return X
end

"""
    change_nfeatures(f, x, n_features)

Change the number of features from `f.n_features` to `n_features`. 

Note that if the input `n_features` is larger than `f.n_features`, it results in
the regeneration of signals based on the current `f.n_features` before 
reselecting the features. This will cause additional features to be less 
accurate and effective.
"""
function change_nfeatures(f::LocalDiscriminantBasis, x::AbstractArray{T,2},
        n_features::Integer) where T<:Number

    @assert size(x,1) == f.n_features
    @assert 1 <= n_features <= f.n

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
