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
    AutocorrelationShell,
    Wavelets, 
    LinearAlgebra, 
    Statistics, 
    Plots

using 
    ..WPD,
    ..SWT,
    ..Utils

# DENOISING
"Relative Error Shrink"
struct RelErrorShrink <: DNFT
    th::Wavelets.Threshold.THType
    t::AbstractFloat
    RelErrorShrink(th, t) = new(th, t)
end

"Stein's Unbiased Risk Estimate (SURE) Shrink"
struct SureShrink <: DNFT 
    th::Wavelets.Threshold.THType
    t::AbstractFloat
    SureShrink(th, t) = new(th, t)
end

"""
    VisuShrink(th, n)

Extension to the `VisuShrink` struct constructor from `Wavelets.jl`.
"""
function Wavelets.Threshold.VisuShrink(th::Wavelets.Threshold.THType, 
    n::Integer)
return VisuShrink(th, sqrt(2*log(n)))
end

"""
    RelErrorShrink([th=HardTH()])

Struct constructor for Relative Error Shrink.
"""
function RelErrorShrink(th::Wavelets.Threshold.THType=HardTH())
    return RelErrorShrink(th, 1.0)
end

"""
    SureShrink(x[, tree=nothing, th=SteinTH()])

Struct constructor for `SureShrink` based on the signal coefficients `x`.
"""
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

Extension of the `denoise` function from `Wavelets.jl`. Denoise a signal of 
input type `inputtype`.

# Arguments:
- `x::AbstractArray{<:Number}`: input signals/coefficients.
- `inputtype::Symbol`: input type of `x`. Current accepted types of inputs are
    - `:sig`: original signals; `x` should be a 2-D array with each column 
        representing a signal.
    - `:dwt`: `dwt`-transformed signal coefficients; `x` should be a 1-D array 
        with each column representing the coefficients of a signal.
    - `:wpt`: `wpt`-transformed signal coefficients; `x` should be a 1-D array 
        with each column representing the coefficients of a signal.
    - `:sdwt`: `sdwt`-transformed signal coefficients; `x` should be a 2-D array
        with each column representing the coefficients of a node.
    - `:swpd`: `swpd`-transformed signal coefficients; `x` should be a 2-D array
        with each column representing the coefficients of a node.
    - `:acwt`: `acwt`-transformed signal coefficients from 
        AutocorrelationShell.jl; `x` should be a 2-D array with each column 
        representing the coefficients of a node.
    - `:acwpt`: `acwpt`-transformed signal coefficients from
        AutocorrelationShell.jl; `x` should be a 2-D array with each column 
        representing the coefficients of a node.
- `wt::Union{DiscreteWavelet, Nothing}`: the discrete wavelet to be used for
    decomposition (for input type `:sig`) and reconstruction. `nothing` can 
    be supplied if no reconstruction is necessary.
- `L::Integer`: the number of decomposition levels. Necessary for input types
    `:sig`, `:dwt`, and `:sdwt`. Default value is set to be 
    `maxtransformlevels(size(x,1))`.
- `tree::BitVector`: the decomposition tree of the signals. Necessary for input
    types `:wpt` and `:swpd`. Default value is set to be 
    `maketree(size(x,1), L, :dwt)`.
- `dnt::DNFT`: denoise type. Default type is set to be `VisuShrink(size(x,1))`.
- `estnoise::Union{Function, Vector{<:Number}}`: noise estimation. Input can be
    provided as a function to estimate noise in signal, or a vector of estimated
    noise. Default is set to be the `noisest` function.
- `smooth::Symbol`: the smoothing method used. `:regular` smoothing thresholds
    all given coefficients, whereas `:undersmooth` smoothing does not threshold
    the lowest frequency subspace node of the wavelet transform. Default is set
    to be `:regular`.

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
    @assert inputtype ∈ [:sig, :dwt, :wpt, :sdwt, :swpd, :acwt, :acwpt]

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
        y = wt === nothing ? x̃ : idwt(x̃, wt, L)

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
        y = wt === nothing ? x̃ : iwpt(x̃, wt, tree) 

    elseif inputtype == :sdwt               # stationary dwt
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, nothing) : estnoise
        # thresholding
        if smooth == :regular
            x̃ = copy(x)
            threshold!(x̃, dnt.th, σ*dnt.t)
        else    # :undersmooth
            temp = x[:,2:end]
            x̃ = [x[:,1] threshold!(temp, dnt.th, σ*dnt.t)]
        end
        # reconstruction
        y = wt === nothing ? x̃ : isdwt(x̃, wt)

    elseif inputtype == :swpd               # stationary wpd
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, tree) : estnoise
        # thresholding
        if smooth == :regular
            leaves = findall(getleaf(tree))
            x̃ = copy(x)
            x̃[:, leaves] = threshold!(x[:,leaves], dnt.th, σ*dnt.t)             
        else    # :undersmooth
            x̃ = copy(x)
            leaves = findall(getleaf(tree))
            _, coarsestnode = coarsestscalingrange(x, tree, true)
            rng = setdiff(leaves, coarsestnode)
            x̃[:,rng] = threshold!(x[:,rng], dnt.th, σ*dnt.t)                    
        end
        # reconstruction
        y = wt === nothing ? x̃ : iswpt(x̃, wt, tree)
    elseif inputtype == :acwt               # autocorrelation dwt
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, nothing) : estnoise
        # thresholding
        if smooth == :regular
            x̃ = copy(x)
            threshold!(x̃, dnt.th, σ*dnt.t)
        else    # :undersmooth
            temp = x[:,2:end]
            x̃ = [x[:,1] threshold!(temp, dnt.th, σ*dnt.t)]
        end
        # reconstruction
        y = iacwt(x̃)

    else                                    # autocorrelation wpt
        @assert ndims(x) > 1
        # noise estimation
        σ = isa(estnoise, Function) ? estnoise(x, true, tree) : estnoise
        # thresholding
        if smooth == :regular
            leaves = findall(getleaf(tree))
            x̃ = copy(x)
            x̃[:, leaves] = threshold!(x[:,leaves], dnt.th, σ*dnt.t)             
        else    # :undersmooth
            x̃ = copy(x)
            leaves = findall(getleaf(tree))
            _, coarsestnode = coarsestscalingrange(x, tree, true)
            rng = setdiff(leaves, coarsestnode)
            x̃[:,rng] = threshold!(x[:,rng], dnt.th, σ*dnt.t)                    
        end
        # reconstruction
        y = iacwpt(x̃, tree)
    end
    return y
end

"""
    denoiseall(x, inputtype, wt[; L=maxtransformlevels(size(x,1)),
        tree=maketree(size(x,1), L, :dwt), dnt=VisuShrink(size(x,1)),
        estnoise=noisest, bestTH=nothing, smooth=:regular])

Denoise multiple signals of input type `inputtype`. 

# Arguments:
- `x::AbstractArray{<:Number}`: input signals/coefficients.
- `inputtype::Symbol`: input type of `x`. Current accepted types of inputs are
    - `:sig`: original signals; `x` should be a 2-D array with each column 
        representing a signal.
    - `:dwt`: `dwt`-transformed signal coefficients; `x` should be a 2-D array 
        with each column representing the coefficients of a signal.
    - `:wpt`: `wpt`-transformed signal coefficients; `x` should be a 2-D array 
        with each column representing the coefficients of a signal.
    - `:sdwt`: `sdwt`-transformed signal coefficients; `x` should be a 3-D array
        with each 2-D slice representing the coefficients of a signal.
    - `:swpd`: `swpd`-transformed signal coefficients; `x` should be a 3-D array
        with each 2-D slice representing the coefficients of a signal.
    - `:acwt`: `acwt`-transformed signal coefficients from
        AutocorrelationShell.jl; `x` should be a 3-D array with each 2-D slice 
        representing the coefficients of a signal.
    - `:acwpt`: `acwpt`-transformed signal coefficients from
        AutocorrelationShell.jl; `x` should be a 3-D array with each 2-D slice 
        representing the coefficients of a signal.
- `wt::Union{DiscreteWavelet, Nothing}`: the discrete wavelet to be used for
    decomposition (for input type `:sig`) and reconstruction. `nothing` can 
    be supplied if no reconstruction is necessary.
- `L::Integer`: the number of decomposition levels. Necessary for input types
    `:sig`, `:dwt`, and `:sdwt`. Default value is set to be 
    `maxtransformlevels(size(x,1))`.
- `tree::BitVector`: the decomposition tree of the signals. Necessary for input
    types `:wpt` and `:swpd`. Default value is set to be 
    `maketree(size(x,1), L, :dwt)`.
- `dnt::DNFT`: denoise type. Default type is set to be `VisuShrink(size(x,1))`.
- `estnoise::Union{Function, Vector{<:Number}}`: noise estimation. Input can be
    provided as a function to estimate noise in signal, or a vector of estimated
    noise. Default is set to be the `noisest` function.
- `bestTH::Union{Function, Nothing}`: method to determine the best threshold 
    value for a group of signals. If `nothing` is given, then each signal will
    be denoised by its respective best threshold value determined from the 
    parameters `dnt` and `estnoise`; otherwise some function can be passed
    to determine the best threshold value from a vector of threshold values, eg:
    `mean` and `median`. Default is set to be `nothing`.
- `smooth::Symbol`: the smoothing method used. `:regular` smoothing thresholds
    all given coefficients, whereas `:undersmooth` smoothing does not threshold
    the lowest frequency subspace node of the wavelet transform. Default is set
    to be `:regular`.

**See alse:** `denoise`, `noisest`
"""
function denoiseall(x::AbstractArray{T1}, 
        inputtype::Symbol,
        wt::Union{DiscreteWavelet,Nothing};
        L::Integer=maxtransformlevels(size(x,1)),
        tree::BitVector=maketree(size(x,1), L, :dwt),
        dnt::S=VisuShrink(size(x,1)),
        estnoise::Union{Function,Vector{T2}}=noisest,   # can be precomputed
        bestTH::Union{Function,Nothing}=nothing,
        smooth::Symbol=:regular) where {T1<:Number, T2<:Number, S<:DNFT}

    @assert inputtype ∈ [:sig, :dwt, :wpt, :sdwt, :swpd, :acwt, :acwpt]
    @assert smooth ∈ [:regular, :undersmooth]
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
"""
    noisest(x, stationary[, tree=nothing])

Extension to the `noisest` function from `Wavelets.jl`. Estimates the noise of
a signal from its coefficients.

# Examples
```julia
x = randn(128)
wt = wavelet(WT.haar)

# noise estimate for dwt transformation
y = dwt(x, wt)
noise = noisest(y, false)

# noise estimate for wpt transformation
tree = maketree(x, :full)
y = wpt(x, wt, tree)
noise = noisest(y, false, tree)

# noise estimate for sdwt transformation
y = sdwt(x, wt)
noise = noisest(y, true)

# noise estimate for swpd transformation
y = swpd(x, wt)
noise = noisest(y, true, tree)
```

**See also:** `relerrorthreshold`
"""
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
    surethreshold(coef, stationary[, tree=nothing])

Determination of the `t` value used for `SureShrink`.

**See also:** `SureShrink`
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
    relerrorthreshold(coef, stationary[, tree, elbows=2; makeplot=false])

Takes in a set of expansion coefficients, 'plot' the threshold vs relative error 
curve and select the best threshold value based on the elbow method.

# Examples
```julia
x = randn(128)
wt = wavelet(WT.haar)

# noise estimate for dwt transformation
y = dwt(x, wt)
noise = relerrorthreshold(y, false)

# noise estimate for wpt transformation
tree = maketree(x, :full)
y = wpt(x, wt, tree)
noise = relerrorthreshold(y, false, tree)

# noise estimate for sdwt transformation
y = sdwt(x, wt)
noise = relerrorthreshold(y, true)

# noise estimate for swpd transformation
y = swpd(x, wt)
noise = relerrorthreshold(y, true, tree)
```

**See also:** `noisest`, `RelErrorShrink`
"""
function relerrorthreshold(coef::AbstractArray{T}, stationary::Bool,
        tree::Union{BitVector,Nothing}=nothing, elbows::Integer=2;
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
        return x[ix[end]]*xmax, p
    else
        return x[ix[end]]*xmax
    end
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