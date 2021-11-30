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

using LinearAlgebra,
      Wavelets,
      Parameters,
      Statistics,
      StatsBase

using ..Utils,
      ..DWT,
      ..BestBasis

include("ldb_energymap.jl")
include("ldb_measures.jl")

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
    f.tree = BestBasis.bestbasis_treeselection(f.cost, f.n, :max)
    Xc = getbasiscoefall(Xw, f.tree)

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
    Xc = getbasiscoefall(Xw, f.tree)
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

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref), [`fit_transform`](@ref),
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
