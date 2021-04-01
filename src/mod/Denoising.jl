module Denoising
export
    # denoising
    RelErrorShrink,
    SureShrink,
    denoiseall,
    surethreshold,
    relerrorthreshold,
    relerrorplot
    
using 
    Wavelets, 
    LinearAlgebra, 
    Statistics, 
    Plots

using 
    ..WPD,
    ..SWT,
    ..Utils

# THRESHOLD
# extension to Wavelets.threshold to allow multiple dimension array input
function Wavelets.threshold(x::AbstractArray{T}, TH::Wavelets.Threshold.THType, 
        t::Real) where T<:Number
    y = Array{T}(undef, size(x))
    return Wavelets.Threshold.threshold!(copyto!(y,x), TH, t)
end

# DENOISING
struct RelErrorShrink <: DNFT   # Relative Error Shrink
    th::Wavelets.Threshold.THType
    t::AbstractFloat
    RelErrorShrink(th, t) = new(th, t)
end
struct SureShrink <: DNFT       # Stein's Unbiased Risk Estimate (SURE) Shrink
    th::Wavelets.Threshold.THType
    t::AbstractFloat
    SureShrink(th, t) = new(th, t)
end

# extension to Wavelet.VisuShrink
function Wavelets.Threshold.VisuShrink(th::Wavelets.Threshold.THType, 
    n::Integer)
return VisuShrink(th, sqrt(2*log(n)))
end

function RelErrorShrink(th::Wavelets.Threshold.THType=HardTH())
    return RelErrorShrink(th, 1.0)
end

function SureShrink(x::AbstractArray{<:Number}, 
        tree::Union{BitVector, Nothing}=nothing,
        th::Wavelets.Threshold.THType=SteinTH())

    t = ndims(x)==1 ? surethreshold(x, false) : surethreshold(x, true, tree)
    return SureShrink(th, t)
end

                                                                                # TODO: add additional threshold determination methods

"""
    denoise(x, inputtype, wt[; L=maxtransformlevels(size(x,1)),
        tree=maketree(size(x,1), L, :dwt), dnt=VisuShrink(size(x,1)),
        estnoise=noisest, smooth=:regular])

Denoise `x` of input type `inputtype` using a discrete wavelet `wt`. Current
accepted input types are:
* `:sig` -> regular signal.
* `:dwt` -> `dwt`-transformed signal. Necessary to provide `L` in this case, 
    otherwise default value of `maxtransformlevels(x)` is assumed.
* `:wpt` -> `wpt`-transformed signal. Necessary to provide `tree` in this case,
    otherwise default value of `maketree(x, :dwt)` (DWT binary tree) is assumed.
* `:sdwt` -> `sdwt`-transformed signal.
* `:swpd` -> `swpd`-transformed signal. Necessary to provide `tree` in this 
    case, otherwise default value of `maketree(x, :dwt)` (DWT binary tree) is 
    assumed.

Additional parameters are the denoising method `dnt`, noise estimation 
`estnoise`, and smooth type `smooth`.
* `dnt` -> takes in values of type `DNFT`. Default value is set to be 
    `VisuShrink`.
* `estnoise` -> takes in either noise estimation functions or numerical values.
    Default value is set to be `noisest`.
* `smooth` -> another denoising method. `:regular` smoothing thresholds all
    expansion coefficients of the signals, while `:undersmooth` does not
    threshold the scaling coefficients in the lowest frequency subspace.
    Default value is set to be `:regular`.

**See also:** `denoiseall`
"""
function Wavelets.Threshold.denoise(x::AbstractArray{T}, 
        inputtype::Symbol,
        wt::Union{DiscreteWavelet,Nothing}; 
        L::Integer=maxtransformlevels(size(x,1)),
        tree::BitVector=maketree(size(x,1), L, :dwt),
        dnt::S=VisuShrink(size(x,1)),
        estnoise::Union{Function,Number}=noisest,   # can be precomputed
        smooth::Symbol=:regular) where {T<:Number, S<:DNFT}

    @assert smooth ∈ [:undersmooth, :regular]
    @assert inputtype ∈ [:sig, :dwt, :wpt, :sdwt, :swpd]

    # wavelet transform if inputtype == :sig
    if inputtype == :sig
        wt === nothing && error("inputtype=:sig not supported with wt=nothing")
        x = dwt(x, wt, L)
        inputtype = :dwt
    end

    if inputtype == :dwt                    # regular dwt
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, false, nothing) : estnoise    
        # thresholding
        if smooth == :regular
            x̃ = threshold(x, dnt.th, σ*dnt.t)
        else    # :undersmooth
            n₀ = nodelength(size(x,1), L)
            x̃ = [x[1:n₀]; threshold(x[(n₀+1):end], dnt.th, σ*dnt.t)]
        end
        # reconstruction
        y = idwt(x̃, wt, L)

    elseif inputtype == :wpt                # regular wpt
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, false, tree) : estnoise
        # thresholding
        if smooth == :regular
            x̃ = threshold(x, dnt.th, σ*dnt.t)
        else    # :undersmooth
            crng = coarsestscalingrange(x, tree, false)
            rng = setdiff(1:size(x,1), crng)
            x̃ = [x[crng]; threshold(x[rng], dnt.th, σ*dnt.t)]
        end
        # reconstruction
        y = iwpt(x̃, wt, tree) 

    elseif inputtype == :sdwt               # stationary dwt
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, nothing) : estnoise
        # thresholding
        if smooth == :regular
            x̃ = threshold(x, dnt.th, σ*dnt.t)
        else    # :undersmooth
            x̃ = [x[:,1] threshold(x[:,2:end], dnt.th, σ*dnt.t)]
        end
        # reconstruction
        y = isdwt(x̃, wt)

    else                                    # stationary wpd
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, tree) : estnoise
        # thresholding
        if smooth == :regular
            leaves = findall(getleaf(tree))
            x̃ = copy(x)
            x̃[:, leaves] = threshold(x[:,leaves], dnt.th, σ*dnt.t)
        else    # :undersmooth
            x̃ = copy(x)
            leaves = findall(getleaf(tree))
            _, coarsestnode = coarsestscalingrange(x, tree, true)
            rng = setdiff(leaves, coarsestnode)
            x̃[:,rng] = threshold(x[:,rng], dnt.th, σ*dnt.t)
        end
        # reconstruction
        y = iswpt(x̃, wt, tree)
    end
    return y
end

function denoiseall(x::AbstractArray{T1}, 
        inputtype::Symbol,
        wt::Union{DiscreteWavelet,Nothing};
        L::Integer=maxtransformlevels(size(x,1)),
        tree::BitVector=maketree(size(x,1), L, :dwt),
        dnt::S=VisuShrink(size(x,1)),
        estnoise::Union{Function,Vector{T2}}=noisest,   # can be precomputed
        bestTH::Union{Function,Nothing}=nothing,
        smooth::Symbol=:undersmooth) where {T1<:Number, T2<:Number, S<:DNFT}

    @assert inputtype ∈ [:sig, :dwt, :wpt, :sdwt, :swpd]
    @assert ndims(x) > 1
    n = size(x, 1)              # signal length
    N = size(x)[end]            # number of signals
    # wavelet transform if inputtype == :sig
    if inputtype == :sig
        wt === nothing && error("inputtype=:sig not supported with wt=nothing")
        x = hcat([dwt(x[:,i], wt, L) for i in 1:N]...)
        inputtype = :dwt
    end

    # denoise each signal
    y = Array{T1}(undef, (n,N))
    if bestTH === nothing           # using individual threshold values
        for i in axes(x, ndims(x))                                              # TODO: figure out a way to make iterating over xw cleaner
            xᵢ = ndims(x) == 2 ? x[:,i] : x[:,:,i]
            # noise estimation
            σ = isa(estnoise, Function) ? estnoise : estnoise[i]
            # denoising
            y[:,i] = denoise(xᵢ, inputtype, wt, L=L, tree=tree, dnt=dnt, 
                estnoise=σ, smooth=smooth)
        end
    else                            # using summary threshold value
        # noise estimation
        if isa(estnoise, Function)
            σ = Vector{AbstractFloat}(undef, N)
            for i in axes(x, ndims(x))
                xᵢ = ndims(x) == 2 ? x[:,i] : x[:,:,i]
                if inputtype == :dwt
                    σ[i] = estnoise(xᵢ, false, nothing)
                elseif inputtype == :wpt
                    σ[i] = estnoise(xᵢ, false, tree)
                elseif inputtype == :sdwt
                    σ[i] = estnoise(xᵢ, true, nothing)
                else
                    σ[i] = estnoise(xᵢ, true, tree)
                end
            end
            σ = bestTH(σ)
        else
            σ = bestTH(estnoise)
        end
        # denoising
        for i in axes(x, ndims(x))
            xᵢ = ndims(x) == 2 ? x[:,i] : x[:,:,i]
            y[:,i] = denoise(xᵢ, inputtype, wt, L=L, tree=tree, dnt=dnt, 
                estnoise=σ, smooth=smooth)
        end
    end
    return y
end

# BEST THRESHOLD VALUES
# extend Wavelets.noisest
function Wavelets.Threshold.noisest(x::AbstractArray{T}, stationary::Bool,
        tree::Union{BitVector,Nothing}=nothing) where T<:Number

    @assert isdyadic(size(x,1))
    if !stationary && isa(tree, Nothing)        # regular dwt
        n₀ = size(x,1) ÷ 2
        dr = x[(n₀+1):end]
    elseif !stationary && isa(tree, BitVector)  # regular wpt
        dr = x[finestdetailrange(x, tree)]
    elseif stationary && isa(tree, Nothing)     # stationary dwt
        dr = x[:,end]
    else                                        # stationary wpd
        dr = x[finestdetailrange(x, tree, true)...]
    end
    return Wavelets.Threshold.mad!(dr)/0.6745
end

"""
    surethreshold()
"""
function surethreshold(coef::AbstractArray{T}, stationary::Bool,
        tree::Union{BitVector,Nothing}=nothing) where T<:Number

    # extract necessary coefficients
    if !stationary                              # dwt or wpt
        y = coef
    elseif stationary && isa(tree, Nothing)     # sdwt
        y = reshape(coef, :)
    else                                        # swpd
        leaves = getleaf(tree)
        y = reshape(coef[:, leaves], :)
    end
    a = sort(abs.(y)).^2
    b = cumsum(a)
    n = length(y)
    c = collect(n-1:-1:0)
    s = b + c .* a
    risk = (n .- (2 * (1:n)) + s)/n
    i = argmin(risk)
    return sqrt(a[i])
end

"""
    relerrorthreshold(coef, stationary[, tree, elbows=2, plotting=false])

Takes in a set of expansion coefficients, 'plot' the threshold vs relative error 
curve and select the best threshold value based on the elbow method.
"""
function relerrorthreshold(coef::AbstractArray{T}, stationary::Bool,
        tree::Union{BitVector,Nothing}=nothing, elbows::Integer=2,
        makeplot::Bool=false) where T<:Number

    @assert elbows >= 1
    # extract necessary coefficients
    if !stationary                              # dwt or wpt
        c = coef
    elseif stationary && isa(tree, Nothing)     # sdwt
        c = reshape(coef, :)
    else                                        # swpd
        leaves = getleaf(tree)
        c = reshape(coef[:, leaves], :)
    end
    # the magnitudes of the coefficients
    x = sort(abs.(c), rev = true)
    # compute the relative error curve
    r = orth2relerror(c)
    # shift the data points
    push!(x, 0)
    pushfirst!(r, r[1])
    # reorder the data points
    xmax = maximum(x)
    ymax = maximum(r)
    x = x[end:-1:1]/xmax
    y = r[end:-1:1]/ymax
    # compute elbow method 
    ix = Vector{Integer}(undef, elbows)
    A = Vector{Vector{T}}(undef, elbows)
    v = Vector{Vector{T}}(undef, elbows)
    ix[1], A[1], v[1] = findelbow(x,y)
    for i in 2:1:elbows
        (ix[i], A[i], v[i]) = findelbow(x[1:ix[i-1]], y[1:ix[i-1]]) 
    end
    # plot relative error curve
    if makeplot
        p = relerrorplot(x*xmax, y*ymax, ix, A, v)
        display(p)
    end
    return x[ix[end]]*xmax
end

"""
    orth2relerror(orth)

Given a vector 'orth' of orthonormal expansion coefficients, return a 
vector of relative approximation errors when retaining the 1,2,...,N 
largest coefficients in magnitude.  
"""
function orth2relerror(orth::AbstractVector{T}) where T<:Number
    # sort the coefficients
    orth = sort(orth.^2, rev = true)
    # compute the relative errors
    return ((abs.(sum(orth) .- cumsum(orth))).^0.5) / sum(orth).^0.5
end

"""
    findelbow(x, y)

Given the x and y coordinates of a curve, return the elbow.  
"""
function findelbow(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Number
    # a unit vector pointing from (x1,y1) to (xN,yN)
    v = [x[end] - x[1], y[end] - y[1]]
    v = v/norm(v,2)
    # subtract (x1,y1) from the coordinates
    xy = [x.-x[1] y.-y[1]]
    # the hypothenuse
    H = reshape((sum(xy.^2, dims = 2)).^0.5, :)
    # the adjacent side
    A = xy * v
    # the opposite side
    O = abs.(H.^2 - A.^2).^0.5       
    # return the largest distance
    return (findmax(O)[2], A, v)
end

# threshold vs relative error curve
function relerrorplot(x::Vector{<:Number}, 
        y::Vector{<:Number}, ix::Vector{<:Integer}, 
        A::Vector{<:Vector{<:Number}}, v::Vector{<:Vector{<:Number}})

    @assert length(ix) == length(A) == length(v)
    elbows = length(ix)
    # rescale x and y values
    xmax = maximum(x)
    ymax = maximum(y)
    # relative error line
    p = plot(x, y, lw = 2, color = :blue, legend = false)
    plot!(p, xlims = (0, 1.004*xmax), ylims = (0, 1.004*ymax))
    for i in 1:elbows
        color = i + 1
        # diagonal line
        endpoint = i > 1 ? ix[i-1] : length(x)
        plot!([x[1], 1.004*x[endpoint]], [y[1], 1.004*y[endpoint]], lw = 2, 
            color = color)
        # perpendicular line
        dropto = [x[1], y[1]] + A[i][ix[i]]*(v[i].*[xmax, ymax])
        plot!(p, [x[ix[i]], dropto[1]], [y[ix[i]], dropto[2]], lw = 2, 
            color = color)
        # highlight point
        scatter!(p, [x[ix[i]]], [y[ix[i]]], color = color)
    end
    # add plot labels
    plot!(p, xlabel = "Threshold", ylabel = "Relative Error")
    return p
end

end # end module