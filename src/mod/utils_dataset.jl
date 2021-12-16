# ========== Structs ==========
"""
    ClassData(type, s₁, s₂, s₃)

Based on the input `type`, generates 3 classes of signals with sample sizes
`s₁`, `s₂`, and `s₃` respectively. Accepted input types are:  
- `:tri`: Triangular signals of length 32
- `:cbf`: Cylinder-Bell-Funnel signals of length 128

Based on N. Saito and R. Coifman in "Local Discriminant Basis and their Applications" in the
Journal of Mathematical Imaging and Vision, Vol. 5, 337-358 (1995).

**See also:** [`generateclassdata`](@ref)
"""
struct ClassData{T<:Integer}
    "Signal type, accepted inputs are `:tri` and `:cbf`"
    type::Symbol
    "Sample size for class 1"
    s₁::T
    "Sample size for class 2"
    s₂::T
    "Sample size for class 3"
    s₃::T
    ClassData(type, s₁, s₂, s₃) = type ∈ [:tri, :cbf] ? new(type, s₁, s₂, s₃) : 
        throw(ArgumentError("Invalid type. Accepted types are :tri and :cbf only."))
end

# ========== Functions ==========
# ----- Signal Generation -----
# Make a set of circularly shifted and noisy signals of original signal.
"""
    duplicatesignals(x, n, k[, noise=false, t=1])

Given a signal `x`, returns `n` shifted versions of the signal, each with shifts
of multiples of `k`. 

Setting `noise = true` allows randomly generated Gaussian noises of μ = 0, 
σ² = t to be added to the circularly shifted signals.

# Arguments
- `x::AbstractVector{T} where T<:Number`: 1D-signal to be duplicated.
- `n::Integer`:: Number of duplicated signals.
- `k::Integer`:: Circular shift size for each duplicated signal.
- `noise::Bool`: (Default: `false`) Whether or not to add Gaussian noise.
- `t::Real`: (Default: 1) Relative size of noise.

# Returns
`::Array{T}`: Duplicated signals.

# Examples
```@repl
using WaveletsExt

x = generatesignals(:blocks)
duplicatesignals(x, 5, 0)      # [x x x x x]
```

*See also:* [`generatesignals`](@ref)
"""
function duplicatesignals(x::AbstractArray{T}, 
                          n::Integer, 
                          k::Integer, 
                          noise::Bool = false, 
                          t::Real = 1) where T<:Number
    sz = size(x)
    N = ndims(x) + 1
    X = Array{T,N}(undef, (sz..., n))
    @inbounds begin
        @views for (i, Xᵢ) in enumerate(eachslice(X, dims=N))
            circshift!(Xᵢ, x, k*(i-1)) 
        end
    end
    X = noise ? X + t*randn(sz...,n) : X
    return X
end

# Generate 6 types of signals used in popular papers.
"""
    generatesignals(fn, L)

Generates a signal of length 2ᴸ given the function symbol `fn`. Current accepted inputs 
below are based on D. Donoho and I. Johnstone in "Adapting to Unknown Smoothness via Wavelet 
Shrinkage" Preprint Stanford, January 93, p 27-28.  
- `:blocks`
- `:bumps`
- `:heavisine`
- `:doppler`
- `:quadchirp`
- `:mishmash`

The code for this function is adapted and translated based on MATLAB's Wavelet Toolbox's 
`wnoise` function.

# Arguments
- `fn::Symbol`: Type of function/signal to generate.
- `L::Integer`: (Default = 7) Size of the signal to generate. Will return a signal of size
  2ᴸ.

# Returns
`::Vector{Float64}`: Signal of length 2ᴸ.

# Examples
```repl
using WaveletsExt

generatesignals(:bumps, 8)
```
"""
function generatesignals(fn::Symbol, L::Integer = 7)
    @assert L >= 1

    t = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
    h = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2]
    
    n = 1<<L
    if fn == :blocks
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j]*(1 .+ sign.(tt .- t[j]))/2)
        end
    elseif fn == :bumps
        h = abs.(h)
        w = 0.01*[0.5, 0.5, 0.6, 1, 1, 3, 1, 1, 0.5, 0.8, 0.5]
        tt = collect(range(0, 1, length=n))
        x = zeros(n)
        for j in eachindex(h)
            x += (h[j] ./ (1 .+ ((tt .- t[j]) / w[j]).^4))
        end
    elseif fn == :heavisine
        x = collect(range(0, 1, length=n))
        x = 4*sin.(4*pi*x) - sign.(x .- 0.3) - sign.(0.72 .- x)
    elseif fn == :doppler
        x = collect(range(0, 1, length=n))
        ϵ = 0.05
        x = sqrt.(x.*(1 .- x)) .* sin.(2*pi*(1+ϵ) ./ (x.+ϵ))
    elseif fn == :quadchirp
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
    elseif fn == :mishmash
        tt = collect(range(0, 1, length=n))
        x = sin.((π/3) * tt .* (n * tt.^2))
        x = x + sin.(π * (n * 0.6902) * tt)
        x = x + sin.(π * tt .* (n * 0.125 * tt))
    else
        throw(ArgumentError("Unrecognised `fn`. Type `?generatefunction` to learn more."))
    end
    return x
end

# Generate 3 classes of signals used in Saito's LDB paper.
"""
    generateclassdata(c[, shuffle])

Generates 3 classes of data given a `ClassData` struct as an input. Returns a matrix 
containing the 3 classes of signals and a vector containing their corresponding labels.

Based on N. Saito and R. Coifman in "Local Discriminant Basis and their Applications" in the
Journal of Mathematical Imaging and Vision, Vol. 5, 337-358 (1995).

# Arguments
- `c::ClassData`: Type of signal classes to generate.
- `shuffle::Bool`: (Default: `true`). Whether or not to shuffle the signals.

# Returns
- `::Matrix{Float64}`: Generated signals.
- `::Vector{Int64}`: Class corresponding to each column of generated signals.

# Examples
```@repl
using WaveletsExt

c = ClassData(:tri, 100, 100, 100)
generateclassdata(c)
```

**See also:** [`ClassData`](@ref)
"""
function generateclassdata(c::ClassData{T}, shuffle::Bool = false) where T<:Integer
    @assert c.s₁ >= 0
    @assert c.s₂ >= 0
    @assert c.s₃ >= 0

    if c.type == :tri
        n = 32
        i = collect(1:n)
        # Define classes
        y = Int64.(vcat(ones(Int, c.s₁), 2*ones(Int, c.s₂), 3*ones(Int, c.s₃)))
        
        # Random value generations
        u₁ = rand(Uniform(0,1), c.s₁)
        u₂ = rand(Uniform(0,1), c.s₂)
        u₃ = rand(Uniform(0,1), c.s₃)
        ϵ₁ = rand(Normal(0,1), (n, c.s₁))
        ϵ₂ = rand(Normal(0,1), (n, c.s₂))
        ϵ₃ = rand(Normal(0,1), (n, c.s₃))

        h₁ = max.(6 .- abs.(i.-7), 0)
        h₂ = max.(6 .- abs.(i.-15), 0)
        h₃ = max.(6 .- abs.(i.-11), 0)
        # Build signals
        H₁ = h₁ * u₁' + h₂ * (1 .- u₁)' + ϵ₁
        H₂ = h₁ * u₂' + h₃ * (1 .- u₂)' + ϵ₂
        H₃ = h₂ * u₃' + h₃ * (1 .- u₃)' + ϵ₃
        
        H = hcat(H₁, H₂, H₃)
    elseif c.type == :cbf
        n = 128
        ϵ = rand(Normal(0,1), (n, c.s₁+c.s₂+c.s₃))
        y = Int64.(vcat(ones(c.s₁), 2*ones(Int, c.s₂), 3*ones(Int, c.s₃)))

        d₁ = DiscreteUniform(16,32)
        d₂ = DiscreteUniform(32,96)

        # cylinder signals
        H₁ = zeros(n,c.s₁)
        a = rand(d₁,c.s₁)
        b = a+rand(d₁,c.s₁)
        η = randn(c.s₁)
        for k in 1:c.s₁
            H₁[a[k]:b[k],k]=(6+η[k])*ones(b[k]-a[k]+1)
        end

        # bell signals
        H₂ = zeros(n,c.s₂)
        a = rand(d₁,c.s₂)
        b = a+rand(d₂,c.s₂)
        η = randn(c.s₂)
        for k in 1:c.s₂
            H₂[a[k]:b[k],k]=(6+η[k])*collect(0:(b[k]-a[k]))/(b[k]-a[k])
        end

        # funnel signals
        H₃ = zeros(n,c.s₃)
        a = rand(d₁,c.s₃)
        b = a+rand(d₂,c.s₃)
        η = randn(c.s₃)
        for k in 1:c.s₃
            H₃[a[k]:b[k],k]=(6+η[k])*collect((b[k]-a[k]):-1:0)/(b[k]-a[k])
        end

        H = hcat(H₁, H₂, H₃) + ϵ
    else
        throw(ArgumentError("Invalid type. Accepted types are :tri and :cbf only."))
    end

    if shuffle
        idx = [1:(c.s₁+c.s₂+c.s₃)...]
        shuffle!(idx)
        H = H[:,idx]
        y = y[idx]
    end

    return H, y
end
