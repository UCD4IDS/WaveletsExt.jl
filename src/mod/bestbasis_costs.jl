## COST COMPUTATION
"""
Cost function abstract type.

**See also:** [`LSDBCost`](@ref), [`JBBCost`](@ref), [`BBCost`](@ref)
"""
abstract type CostFunction end

"""
    LSDBCost <: CostFunction

Cost function abstract type specifically for LSDB.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`BBCost`](@ref)
"""
abstract type LSDBCost <: CostFunction end

"""
    JBBCost <: CostFunction
    
Cost function abstract type specifically for JBB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref), [`BBCost`](@ref)
"""
abstract type JBBCost <: CostFunction end

"""
    BBCost <: CostFunction
    
Cost function abstract type specifically for BB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref), [`JBBCost`](@ref)
"""
abstract type BBCost <: CostFunction end

@doc raw"""
    LoglpCost <: JBBCost
    
``\log \ell^p`` information cost used for JBB. Typically, we set `p=2` as in 
Wickerhauser's original algorithm.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`NormCost`](@ref)
"""
@with_kw struct LoglpCost{T<:Real} <: JBBCost 
    p::T = 2
end

@doc raw"""
    NormCost <: JBBCost
    
``p``-norm information cost used for JBB.

**See also:** [`CostFunction`](@ref), [`JBBCost`](@ref), [`LoglpCost`](@ref)
"""
@with_kw struct NormCost <: JBBCost 
    p::Number = 1
end

"""
    DifferentialEntropyCost <: LSDBCost

Differential entropy cost used for LSDB.

**See also:** [`CostFunction`](@ref), [`LSDBCost`](@ref)
"""
struct DifferentialEntropyCost <: LSDBCost end

"""
    ShannonEntropyCost <: LSDBCost

Shannon entropy cost used for BB.

**See also:** [`CostFunction`](@ref), [`BBCost`](@ref), 
    [`LogEnergyEntropyCost`](@ref)
"""
struct ShannonEntropyCost <: BBCost end

"""
    LogEnergyEntropyCost <: LSDBCost

Log energy entropy cost used for BB.

**See also:** [`CostFunction`](@ref), [`BBCost`](@ref), 
    [`ShannonEntropyCost`](@ref)
"""
struct LogEnergyEntropyCost <: BBCost end

## ----- COST COMPUTATION -----
# Cost functions for individual best basis algorithm (Same as Wavelets.jl)
"""
    coefcost(x, et[, nrm])

# Arguments
- `x::AbstractArray{T} where T<:AbstractFloat`: An array of values to compute the cost.
- `et::CostFunction`: Type of cost function.
- `nrm::T where T<:AbstractFloat`: The norm of the `x`. Only applicable when `et` is a
  `BBCost`.

# Returns
- `::T`: Cost of `x`.

**See also:** [`bestbasistree`](@ref)
"""
function coefcost(x::T, et::ShannonEntropyCost, nrm::T) where T<:AbstractFloat
    s = (x/nrm)^2
    c = s == 0.0 ? -zero(T) : -s*log(s)
    return c
end

function coefcost(x::T, et::LogEnergyEntropyCost, nrm::T) where T<:AbstractFloat
    s = (x/nrm)^2
    c = s == 0.0 ? -zero(T) : -log(s)
    return c
end

function coefcost(x::AbstractArray{T}, et::BBCost, nrm::T=norm(x)) where T<:AbstractFloat
    @assert nrm ≥ 0
    sum = zero(T)
    nrm == sum && return sum
    for i in eachindex(x)
        @inbounds sum += coefcost(x[i], et, nrm)
    end
    return sum
end

# Cost functions for joint best basis (JBB)
function coefcost(x::AbstractArray{T}, et::LoglpCost) where T<:Number
    xₐ = abs.(x)
    return et.p * sum(log.(xₐ))
end

coefcost(x::AbstractArray{T}, et::NormCost) where T<:Number = norm(x, et.p)^et.p

# Cost functions for least statistically dependent basis (LSDB)
function coefcost(x::AbstractVector{T}, et::DifferentialEntropyCost) where 
                  T<:AbstractFloat
    N = length(x)                           # length of vector x
    M = 50                                  # arbitrary large number M

    nbins = ceil(Int64, (30 * N)^(1/5))     # number of bins per histogram
    mbins = ceil(Int64, M/nbins)            # number of histograms calculated

    σ = std(x)                              # standard deviation of x
    # setup range for ASH
    s = 0.5                     
    δ = (maximum(x) - minimum(x) + σ)/((nbins+1)*mbins-1)       
    rng = (minimum(x) - s*σ):δ:(maximum(x) + s*σ)     
    # compute ASH          
    epdf = ash(x, rng=rng, m=mbins, kernel=Kernels.triangular)
    ent = 0
    for k in 1:N
        ent -= (1/N) * log(AverageShiftedHistograms.pdf(epdf, x[k]))
    end
    return ent
end

function coefcost(x::AbstractArray{T}, et::DifferentialEntropyCost) where T<:Number
    cost = 0
    sz = size(x)[1:end-1] |> prod
    for i in 1:sz
        @inbounds cost += coefcost(x[i:sz:end], DifferentialEntropyCost())
    end
    return cost
end