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
    fitdec!,
    fit_transform,
    transform,
    inverse_transform,
    change_nfeatures

using LinearAlgebra,
      Wavelets,
      Statistics
import Parameters: @with_kw
import StatsBase: mad

using ..Utils,
      ..DWT
import ..BestBasis: bestbasis_treeselection

include("ldb_energymap.jl")
include("ldb_measures.jl")

## LOCAL DISCRIMINANT BASIS
"""
    LocalDiscriminantBasis

Class type for the Local Discriminant Basis (LDB), a feature selection algorithm developed
by N. Saito and R. Coifman in "Local Discriminant Bases and Their Applications" in the
Journal of Mathematical Imaging and Vision, Vol 5, 337-358 (1995). This struct contains the
following field values: 

# Parameters and Attributes:
- `wt::DiscreteWavelet`: (Default: `wavelet(WT.haar)`) A discrete wavelet for transform
  purposes.
- `max_dec_level::Union{Integer, Nothing}`: (Default: `nothing`) Max level of wavelet packet
  decomposition to be computed.
- `dm::DiscriminantMeasure`: (Default: `AsymmetricRelativeEntropy()`) the discriminant
    measure for the LDB algorithm. Supported measures are the `AsymmetricRelativeEntropy()`,
    `LpDistance()`, `SymmetricRelativeEntropy()`, and `HellingerDistance()`.
- `en::EnergyMap`: (Default: `TimeFrequency()`) the type of energy map used. Supported maps
    are `TimeFrequency()`, `ProbabilityDensity()`, and `Signatures()`.
- `dp::DiscriminantPower()`: (Default: `BasisDiscriminantMeasure()`) the measure of
    discriminant power among expansion coefficients. Supported measures are
    `BasisDiscriminantMeasure()`, `FishersClassSeparability()`, and
    `RobustFishersClassSeparability()`. 
- `top_k::Union{Integer, Nothing}`: (Default: `nothing`) the top-k coefficients used in each
    node to determine the discriminant measure.
- `n_features::Union{Integer, Nothing}`: (Default: `nothing`) the dimension of output after
    undergoing feature selection and transformation.
- `sz::Union{Vector{T} where T<:Integer, Nothing}`: (Default: `nothing`) Size of signal
- `Γ::Union{AbstractArray{<:AbstractFloat}, AbstractArray{NamedTuple{(:coef, :weight),
  Tuple{S1, S2}}} where {S1<:Array{T} where T<:AbstractFloat, S2<:Union{T, Array{T}} where
  T<:AbstractFloat}, Nothing}`: (Default: `nothing`) computed energy map
- `DM::Union{AbstractArray{<:AbstractFloat}, Nothing}`: (Default: `nothing`) computed
  discriminant measure
- `cost::Union{AbstractVector{<:AbstractFloat}, Nothing}`: (Default: `nothing`) computed
    wavelet packet decomposition (WPD) tree cost based on the discriminant measure `DM`.
- `tree::Union{BitVector, Nothing}`: (Default: `nothing`) computed best WPD tree based on
    the discriminant measure `DM`.
- `DP::Union{AbstractVector{<:AbstractFloat}, Nothing}`: (Default: `nothing`) computed
  discriminant power
- `order::Union{AbstractVector{Integer}, Nothing}`: (Default: `nothing`) ordering of `DP` by
  descending order.
"""
@with_kw mutable struct LocalDiscriminantBasis
    # to be declared by user
    wt::DiscreteWavelet = wavelet(WT.haar)
    max_dec_level::Union{Integer, Nothing} = nothing
    dm::DiscriminantMeasure = AsymmetricRelativeEntropy()
    en::EnergyMap = TimeFrequency()
    dp::DiscriminantPower = BasisDiscriminantMeasure()
    top_k::Union{Integer, Nothing} = nothing
    n_features::Union{Integer, Nothing} = nothing
    # to be computed in fit! or fitdec!
    sz::Union{Tuple{Vararg{T,N}} where {T<:Integer,N}, Nothing} = nothing
    Γ::Union{AbstractArray{<:AbstractFloat}, 
             AbstractArray{NamedTuple{(:coef, :weight), Tuple{S1, S2}}} where
                {S1<:Array{T} where T<:AbstractFloat,
                 S2<:Union{AbstractFloat, Array{<:AbstractFloat}}},
             Nothing} = nothing
    DM::Union{AbstractArray{<:AbstractFloat}, Nothing} = nothing
    cost::Union{AbstractVector{<:AbstractFloat}, Nothing} = nothing
    tree::Union{BitVector, Nothing} = nothing
    DP::Union{AbstractArray{<:AbstractFloat}, Nothing} = nothing
    order::Union{AbstractVector{Integer}, Nothing} = nothing
end

# """
#     LocalDiscriminantBasis([; 
#         wt=wavelet(WT.haar),
#         max_dec_level=nothing,
#         dm=AsymmetricRelativeEntropy(), em=TimeFrequency(), 
#         dp=BasisDiscriminantMeasure(), top_k=nothing,
#         n_features=nothing]
#     )

# Class constructor for `LocalDiscriminantBasis`. 

# # Arguments:
# - `wt::DiscreteWavelet`: Wavelet used for decomposition of signals. Default is
#     set to be `wavelet(WT.haar)`.
# - `max_dec_level::Union{Integer, Nothing}`: max level of wavelet packet
#     decomposition to be computed. When `max_dec_level=nothing`, the maximum
#     transform levels will be used. Default is set to be `nothing`.
# - `dm::DiscriminantMeasure`: the discriminant measure for the LDB algorithm. 
#     Supported measures are the `AsymmetricRelativeEntropy()`, `LpDistance()`, 
#     `SymmetricRelativeEntropy()`, and `HellingerDistance()`. Default is set to
#     be `AsymmetricRelativeEntropy()`.
# - `en::EnergyMap`: the type of energy map used. Supported maps are 
#     `TimeFrequency()` and `ProbabilityDensity()`. Default is set to be 
#     `TimeFrequency()`.
# - `dp::DiscriminantPower=BasisDiscriminantMeasure()`: the measure of 
#     discriminant power among expansion coefficients. Supported measures are 
#     `BasisDiscriminantMeasure()`, `FishersClassSeparability()`, and 
#     `RobustFishersClassSeparability()`. Default is set to be `BasisDiscriminantMeasure()`.
# - `top_k::Union{Integer, Nothing}`: the top-k coefficients used in each node to 
#     determine the discriminant measure. When `top_k=nothing`, all coefficients 
#     are used to determine the discriminant measure. Default is set to be 
#     `nothing`.
# - `n_features::Union{Integer, Nothing}`: the dimension of output after 
#     undergoing feature selection and transformation. When `n_features=nothing`,
#     all features will be returned as output. Default is set to be `nothing`.
# """
# function LocalDiscriminantBasis(; wt::DiscreteWavelet=wavelet(WT.haar),
#         max_dec_level::Union{Integer, Nothing}=nothing, 
#         dm::DiscriminantMeasure=AsymmetricRelativeEntropy(),
#         en::EnergyMap=TimeFrequency(), 
#         dp::DiscriminantPower=BasisDiscriminantMeasure(), 
#         top_k::Union{Integer, Nothing}=nothing, 
#         n_features::Union{Integer, Nothing}=nothing)

#     return LocalDiscriminantBasis(
#         wt, max_dec_level, dm, en, dp, top_k, n_features, 
#         nothing, nothing, nothing, nothing, nothing, nothing, nothing
#     )
# end

"""
    fit!(f, X, y)

Fits the Local Discriminant Basis feature selection algorithm `f` onto the 
signals `X` (or the decomposed signals `Xw`) with labels `y`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit_transform`](@ref),
    [`transform`](@ref), [`inverse_transform`](@ref), [`change_nfeatures`](@ref)
"""
function fit!(f::LocalDiscriminantBasis, X::AbstractArray{S}, y::AbstractVector{T}) where 
             {S<:AbstractFloat, T}

    # basic summary of data
    @assert 2 ≤ ndims(X) ≤ 3
    sz = size(X)[1:end-1]
    L = maxtransformlevels(min(sz...))
    # change LocalDiscriminantBasis parameters if necessary
    f.max_dec_level = isnothing(f.max_dec_level) ? L : f.max_dec_level
    @assert 1 ≤ f.max_dec_level ≤ L
    
    # wavelet packet decomposition
    Xw = wpdall(X, f.wt, f.max_dec_level)

    # fit local discriminant basis
    fitdec!(f, Xw, y)
    return nothing
end

function fitdec!(f::LocalDiscriminantBasis, Xw::AbstractArray{S}, y::AbstractVector{T}) where 
                {S<:AbstractFloat, T}
    # basic summary of data
    @assert 3 ≤ ndims(Xw) ≤ 4
    c = unique(y)       # unique classes
    nc = length(c)      # number of classes
    Ny = length(y)
    f.sz = size(Xw)[1:end-2]
    L = size(Xw)[end-1]
    Nx = size(Xw)[end]
    nelem = prod(f.sz)  # total number of elements in each signal

    # change LocalDiscriminantBasis parameters if necessary
    f.top_k = isnothing(f.top_k) ? nelem : f.top_k
    f.n_features = isnothing(f.n_features) ? nelem : f.n_features
    f.max_dec_level = isnothing(f.max_dec_level) ? L-1 : f.max_dec_level

    # parameter checking
    @assert Nx == Ny
    @assert 1 ≤ f.top_k ≤ nelem
    @assert 1 ≤ f.n_features ≤ nelem
    @assert f.max_dec_level+1 == L
    @assert 1 ≤ f.max_dec_level ≤ maxtransformlevels(min(f.sz...))
    @assert nc > 1

    # --- Construct energy map for each class ---
    f.Γ = energy_map(Xw, y, f.en)

    # --- Compute discriminant measure D and obtain tree cost ---
    f.DM = discriminant_measure(f.Γ, f.dm)
    tree_type = ndims(Xw) == 3 ? :binary : :quad
    cost_len = ndims(Xw) == 3 ? gettreelength(1<<L) : gettreelength(1<<L,1<<L)
    f.cost = Vector{S}(undef, cost_len)
    for i in eachindex(f.cost)
        d = getdepth(i, tree_type)    # node level
        # Extract coefficients for current node
        if tree_type == :binary
            θ = i - 1<<d
            nθ = nodelength(f.sz..., d)         # Number of elements in current node
            rng = (θ*nθ+1):((θ+1)*nθ)
            DMθ = f.DM[rng,d+1]
        else
            rng₁ = getrowrange(f.sz[1], i)
            rng₂ = getcolrange(f.sz[2], i)
            nθ = length(rng₁) * length(rng₂)    # Number of elements in current node
            DMθ = f.DM[rng₁,rng₂,d+1] |> vec
        end
        # Compute cost
        if f.top_k < nθ
            sort!(DMθ, rev=true)
            f.cost[i] = sum(DMθ[1:f.top_k])
        else
            f.cost[i] = sum(DMθ)
        end
    end

    # --- Select best tree and best set of expansion coefficients ---
    f.tree = bestbasis_treeselection(f.cost, f.sz..., :max)
    Xc = getbasiscoefall(Xw, f.tree)

    # --- Obtain and order basis functions by power of discrimination ---
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
function transform(f::LocalDiscriminantBasis, X::AbstractArray{T}) where T
    # check necessary measurements
    sz = size(X)[1:end-1]
    N = size(X)[end]
    nelem = prod(f.sz)  # total number of elements in each signal
    @assert 2 ≤ ndims(X) ≤ 3
    @assert !isnothing(f.max_dec_level)
    @assert !isnothing(f.top_k)
    @assert !isnothing(f.n_features)
    @assert !isnothing(f.sz)
    @assert !isnothing(f.Γ)
    @assert !isnothing(f.DM)
    @assert !isnothing(f.cost)
    @assert !isnothing(f.tree)
    @assert !isnothing(f.DP)
    @assert !isnothing(f.order)
    @assert sz == f.sz
    @assert 1 ≤ f.max_dec_level ≤ maxtransformlevels(min(f.sz...))
    @assert 1 ≤ f.top_k ≤ nelem
    @assert 1 ≤ f.n_features ≤ nelem

    # wpt on X based on given f.tree
    Xw = wptall(X, f.wt, f.tree)
    # Extract best features
    Xc = Array{T,2}(undef, f.n_features, N)
    for (i, xw) in enumerate(eachslice(Xw, dims=ndims(Xw)))
        Xc[:,i] = xw[f.order[1:f.n_features]]
    end
    return Xc
end

"""
    fit_transform(f, X, y)

Fit and transform the signals `X` with labels `y` based on the LDB class `f`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref),
    [`transform`](@ref), [`inverse_transform`](@ref), [`change_nfeatures`](@ref)
"""
function fit_transform(f::LocalDiscriminantBasis, 
                       X::AbstractArray{S}, 
                       y::AbstractVector{T}) where {S<:AbstractFloat, T}
    # get necessary measurements
    @assert 2 ≤ ndims(X) ≤ 3
    sz = size(X)[1:end-1]
    N = size(X)[end]
    f.max_dec_level = isnothing(f.max_dec_level) ? maxtransformlevels(min(sz...)) : f.max_dec_level
    @assert 1 ≤ f.max_dec_level ≤ maxtransformlevels(min(sz...))

    # wpd on X
    Xw = wpdall(X, f.wt, f.max_dec_level)

    # fit LDB and return best features
    fitdec!(f, Xw, y)
    Xw = getbasiscoefall(Xw, f.tree)
    # Extract best features
    Xc = Array{S,2}(undef, f.n_features, N)
    for (i, xw) in enumerate(eachslice(Xw, dims=ndims(Xw)))
        Xc[:,i] = xw[f.order[1:f.n_features]]
    end
    return Xc
end

"""
    inverse_transform(f, X)

Compute the inverse transform on the feature matrix `x` to form the original
signal based on the LDB class `f`.

**See also:** [`LocalDiscriminantBasis`](@ref), [`fit!`](@ref),
    [`fit_transform`](@ref), [`transform`](@ref), [`change_nfeatures`](@ref)
"""
function inverse_transform(f::LocalDiscriminantBasis, X::AbstractArray{T,2}) where 
                           T<:AbstractFloat
    # get necessary measurements
    @assert size(X,1) == f.n_features
    N = size(X,2)

    # insert the features x into the full matrix X padded with 0's
    Xc = zeros(T, (f.sz..., N))
    @views for (xc, x) in zip(eachslice(Xc, dims=ndims(Xc)), eachslice(X, dims=2))
        for (i,j) in enumerate(f.order[1:f.n_features])
            xc[j] = x[i]
        end
    end

    # iwpt on X
    Xₜ = iwptall(Xc, f.wt, f.tree)
    return Xₜ
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
    @assert !isnothing(f.n_features)
    @assert size(x,1) == f.n_features || throw(ArgumentError("f.n_features and number of rows of x do not match!"))
    @assert 1 ≤ n_features ≤ prod(f.sz)

    # change number of features
    if f.n_features ≥ n_features
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
