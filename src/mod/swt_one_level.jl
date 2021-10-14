# ========== Single Step Stationary Wavelet Transform ==========
"""
    sdwt_step(v, d, h, g)

Perform one level of the stationary discrete wavelet transform (SDWT) on the vector `v`,
which is the `d`-th level scaling coefficients (Note the 0th level scaling coefficients is
the raw signal). The vectors `h` and `g` are the detail and scaling filters.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
SWT.sdwt_step(v, 0, h, g)
```

**See also:** [`sdwt_step!`](@ref)
"""
function sdwt_step(v::AbstractVector{T}, d::Integer, h::Array{S,1}, g::Array{S,1}) where 
                  {T<:Number, S<:Number}
    n = length(v)
    w₁ = zeros(T, n)
    w₂ = zeros(T, n)

    sdwt_step!(w₁, w₂, v, d, h, g)
    return w₁, w₂
end

"""
    sdwt_step!(w₁, w₂, v, d, h, g)

Same as `sdwt_step` but without array allocation.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `v::AbstractVector{T} where T<:Number`: Vector of coefficients from a node at level `d`.
- `d::Integer`: Depth level of `v`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `w₁::Vector{T}`: Output from the low pass filter.
- `w₂::Vector{T}`: Output from the high pass filter.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)
w₁ = zeros(8)
w₂ = zeros(8)

# One step of SDWT
SWT.sdwt_step!(w₁, w₂, v, 0, h, g)
```

**See also:** [`sdwt_step`](@ref)
"""
function sdwt_step!(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T},
                    v::AbstractVector{T},
                    d::Integer, 
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    # Sanity check
    @assert length(w₁) == length(w₂) == length(v)
    @assert length(h) == length(g)
    
    # Setup
    n = length(v)                   # Signal length
    filtlen = length(h)             # Filter length

    # One step of stationary transform
    for i in 1:n
        k₁ = mod1(i-(1<<d),n)       # Start index for low pass filtering
        k₂ = i                      # Start index for high pass filtering
        @inbounds w₁[i] = g[end] * v[k₁]
        @inbounds w₂[i] = h[1] * v[k₂]
        for j in 2:filtlen
            k₁ = k₁+(1<<d) |> k₁ -> k₁>n ? mod1(k₁,n) : k₁
            k₂ = k₂-(1<<d) |> k₂ -> k₂≤0 ? mod1(k₂,n) : k₂
            @inbounds w₁[i] += g[end-j+1] * v[k₁]
            @inbounds w₂[i] += h[j] * v[k₂]
        end
    end
    return w₁, w₂
end

"""
    isdwt_step(w₁, w₂, d, h, g)
    isdwt_step(w₁, w₂, d, sv, sw, h, g)

Perform one level of the inverse stationary discrete wavelet transform (ISDWT) on the
vectors `w₁` and `w₂`, which are the `d+1`-th level scaling coefficients (Note the 0th level
scaling coefficients is the raw signal). The vectors `h` and `g` are the detail and scaling
filters.

!!! note
    One can decide to choose the average-based inverse transform or the shift-based inverse
    transform. For shift based, one needs to specify the shifts of the parent and children
    nodes; for the average based, the output is the average of all possible shift-based
    inverse transform outputs.

# Arguments
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
v̂ = SWT.isdwt_step(w₁, w₂, 0, h, g)         # Average based
ṽ = SWT.isdwt_step(w₁, w₂, 0, 0, 1, h, g)   # Shift based
```

**See also:** [`isdwt_step!`](@ref)
"""
function isdwt_step(w₁::AbstractVector{T},
                    w₂::AbstractVector{T},
                    d::Integer,
                    h::Array{S,1},
                    g::Array{S,1}) where {T<:Number, S<:Number}
    v = similar(w₁)
    isdwt_step!(v, w₁, w₁, d, h, g)
    return v
end

function isdwt_step(w₁::AbstractVector{T}, 
                    w₂::AbstractVector{T}, 
                    d::Integer, 
                    sv::Integer, 
                    sw::Integer, 
                    h::Array{S,1}, 
                    g::Array{S,1}) where {T<:Number, S<:Number}
    v = similar(w₁)
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g)
    return v
end

"""
    isdwt_step!(v, w₁, w₂, d, h, g)
    isdwt_step!(v, w₁, w₂, d, sv, sw, h, g[; add2out])

Same as `isdwt_step` but without array allocation.

# Arguments
- `v::AbstractVector{T} where T<:Number`: Vector allocation for reconstructed coefficients.
- `w₁::AbstractVector{T} where T<:Number`: Vector allocation for output from low pass
  filter.
- `w₂::AbstractVector{T} where T<:Number`: Vector allocation for output from high pass
  filter.
- `d::Integer`: Depth level of parent node of `w₁` and `w₂`.
- `sv::Integer`: Shift of parent node `v`.
- `sw::Integer`: Shift of children nodes `w₁` and `w₂`.
- `h::Vector{S} where S<:Number`: High pass filter.
- `g::Vector{S} where S<:Number`: Low pass filter.

# Keyword Arguments
- `add2out::Bool`: (Default: `false`) Whether to add computed result directly to output `v`
  or rewrite computed result to `v`.

# Returns
- `v::Vector{T}`: Reconstructed coefficients.

# Examples
```julia
using Wavelets, WaveletsExt

# Setup
v = randn(8)
v̂ = similar(v)
ṽ = similar(v)
wt = wavelet(WT.haar)
g, h = WT.makereverseqmfpair(wt, true)

# One step of SDWT
w₁, w₂ = SWT.sdwt_step(v, 0, h, g)

# One step of ISDWT
SWT.isdwt_step!(v̂, w₁, w₂, 0, h, g)          # Average based
SWT.isdwt_step!(ṽ, w₁, w₂, 0, 0, 1, h, g)    # Shift based
```

**See also:** [`isdwt_step`](@ref)
"""
function isdwt_step!(v::AbstractVector{T},
                     w₁::AbstractVector{T},
                     w₂::AbstractVector{T},
                     d::Integer,
                     h::Array{S,1},
                     g::Array{S,1}) where {T<:Number, S<:Number}
    nd = 1 << d             # Number of blocks for parent node

    # isdwt_step for each shift
    for sv in 0:(nd-1)
        sw₁ = sv            # Shift of w₁, w₂ with no additional shift
        sw₂ = sv + 1<<d     # Shift of w₁, w₂ with addtional shift
        isdwt_step!(v, w₁, w₂, d, sv, sw₁, h, g)                 # Without shift
        isdwt_step!(v, w₁, w₂, d, sv, sw₂, h, g, add2out=true)   # With shift
    end
    
    # Get average output of shift vs no shift
    for i in eachindex(v)
        @inbounds v[i] /= 2
    end
end

function isdwt_step!(v::AbstractVector{T}, 
                     w₁::AbstractVector{T}, 
                     w₂::AbstractVector{T}, 
                     d::Integer, 
                     sv::Integer, 
                     sw::Integer,
                     h::Array{S,1}, 
                     g::Array{S,1};
                     add2out::Bool = false) where {T<:Number, S<:Number}
    # Sanity check
    @assert sw ≥ sv

    # Setup
    n = length(v)               # Signal length
    filtlen = length(h)         # filter length
    ip = sv + 1                 # Parent start index
    sp = 1 << d                 # Parent step size
    ic = sw + 1                 # Child start index
    sc = 1 << (d+1)             # Child step size
    
    # One step of inverse SDWT
    for (t, m) in enumerate(ip:sp:n)
        i₀ = mod1(t,2)                                  # Pivot point for filter
        i₁ = filtlen-i₀+1                               # Index for low pass filter g
        i₂ = mod1(t+1,2)                                # Index for high pass filter h
        j = sw==sv ? mod1(m-1<<d,n) : mod1(m+sp-1<<d,n) # Position of v, shift needed if sw≠sv
        k₁ = ((t-1)>>1) * sc + ic                       # Index for approx coefs w₁
        k₂ = ((t-1)>>1) * sc + ic                       # Index for detail coefs w₂
        @inbounds v[j] = add2out ? v[j]+g[i₁]*w₁[k₁]+h[i₂]*w₂[k₂] : g[i₁]*w₁[k₁]+h[i₂]*w₂[k₂]
        for i in (i₀+2):2:filtlen
            i₁ = filtlen-i+1
            i₂ = i + isodd(i) - iseven(i)
            k₁ = k₁-(1<<(d+1)) |> k₁ -> k₁≤0 ? mod1(k₁,n) : k₁
            k₂ = k₂+(1<<(d+1)) |> k₂ -> k₂>n ? mod1(k₂,n) : k₂
            @inbounds v[j] += g[i₁]*w₁[k₁]+h[i₂]*w₂[k₂]
        end
    end
    return v
end
