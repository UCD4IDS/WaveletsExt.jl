module Visualizations
export 
    plot_tfbdry,
    wiggle,
    wiggle!

using
    Plots,
    Statistics
using
    ..Utils

"""
    treenodes_matrix(x) 

Given a `BitVector` of nodes in a binary tree, output the matrix representation of the
nodes.

# Arguments
- `x::BitVector`: BitVector representing a binary tree, where an input is 1 if the
  corresponding node exists and has children, and 0 if the corresponding node does not
  exist/does not have children.

# Returns
`::BitMatrix`: BitMatrix representation of the tree, where each column corresponds to a
level of the binary tree. The inputs in the matrix are 1 if the corresponding node exists
and has children, and 0 if the corresponding node does not exist/does not have children.

# Visualization
```
# Binary tree
      x
     / \\
    x   o
   / \\
  x   o
 / \\
o   o

# BitVector representation
[1,1,0,1,0,0,0]

# BitMatrix representation
[1 1 1 1;
 1 1 0 0;
 1 0 0 0]
```

# Examples
```@repl
using Wavelets, WaveletsExt

tree = maketree(8, 3, :dwt)
WaveletsExt.Visualizations.treenodes_matrix(tree)
```
"""
function treenodes_matrix(x::BitVector)  
    n = (length(x) + 1) >> 1
    L = floor(Integer, log2(n))

    result = falses(L+1, n)
    i = 1
    for lvl in 0:L
        c = 1 << lvl                  # number of nodes at current level
        n₀ = nodelength(n, lvl)       # length of nodes at current level
        for node in 0:(c - 1)
            rng = (node * n₀ + 1):((node + 1) * n₀)
            result[lvl+1, rng] = repeat([x[i]], n₀)
            i += 1
        end
    end

    return result
end

"""
    plot_tfbdry(tree[; start, nd_col, ln_col, bg_col])

Given a tree, output a visual representation of the leaf nodes, user will have the option to
start the node count of each level with 0 or 1.

# Arguments
- `tree::BitVector`: Tree for plotting the leaf nodes. Comes in the form of a `BitVector`.
- `start::Integer`: (Default: `0`) Whether to zero-index or one-index the root of the tree.
- `nd_col::Symbol`: (Default: `:white`) Color of the leaf nodes.
- `ln_col::Symbol`: (Default: `:white`) Color of lines in plot.
- `bg_col::Symbol`: (Default: `:black`) Color of background.

# Returns
`::Plots.Plot`: Plot object with the visual representation of the leaf nodes.

# Examples
```julia
using Wavelets, WaveletsExt

# Build a tree using Wavelets `maketree`
tree = maketree(128, 7, :dwt)

# Plot the leaf nodes
plot_tfbdry(tree)
```
"""
function plot_tfbdry(tree::BitVector;
                     depth::Integer = log2(length(tree)+1), 
                     start::Integer = 0, 
                     nd_col::Symbol = :white,
                     ln_col::Symbol = :white,
                     bg_col::Symbol = :black)

    @assert 0 <= start <= 1
    leaf = getleaf(tree)

    ncol = 1 << (depth-1)
    nrow = depth
    mat = treenodes_matrix(leaf[1:(1<<depth-1)])

    p = heatmap(0:(ncol-1), start:(nrow+start-1), mat, color = [bg_col, nd_col], 
                legend = false, background_color = bg_col)

    plot!(p, xlims = (start-0.5, ncol+start-0.5), ylims = (-0.5, nrow-0.5), yticks = 0:nrow)
    # plot horizontal lines
    for j in 0:(nrow-1)
        x_rng = (start-0.5):(ncol+start-0.5)
        y_val = (j-0.5)*ones(ncol+1)
        plot!(p, x_rng, y_val, color = ln_col, legend = false)
    end

    # plot vertical lines
    for j in 1:(nrow-1)
        for jj in 1:2^(j-1)
            x_val = (ncol/2^j)*(2*jj-1)+start-0.5;
            y_rng = j-0.5:nrow-0.5
            plot!(p, x_val*ones(nrow-j+1), y_rng, color = ln_col)
        end
    end
    plot!(p, (ncol+start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = ln_col)
    plot!(p, (start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = ln_col)
    plot!(p, yaxis = :flip)

    return p
end

"""
    wiggle(wav[; taxis, zaxis, sc, EdgeColor, FaceColor, Orient, Overlap, ZDir])
    wiggle(plt, wav[; taxis, zaxis, sc, EdgeColor, FaceColor, Orient, Overlap, ZDir])

Plots a set of shaded wiggles.

# Arguments
- `plt::Plots.Plot`: Input plot to plot shaded wiggles.
- `wav::AbstractArray{<:Number,2}`: Matrix of waveform columns.

# Keyword Arguments
- `taxis::AbstractVector`: (Default: `1:size(wav,1)`) Time axis vector
- `zaxis::AbstractVector`: (Default: `1:size(wav,2)`) Space axis vector
- `sc::Real`: (Default: `1`) Scale factor/magnification.
- `EdgeColor::Symbol`: (Default: `:black`) Sets edge of wiggles color.
- `FaceColor::Symbol`: (Default: `:black`) Sets shading color of wiggles.
- `Overlap::Bool`: (Default: `true`) How signals are scaled.
    - `true`  - Signals overlap (default);
    - `false` - Signals are scaled so they do not overlap.
- `Orient::Symbol`: (Default: `:across`) Controls orientation of wiggles.
    - `:across` - from left to right
    - `:down`   - from top to down
- `ZDir::Symbol`: (Default: `:normal`) Direction of space axis.
    - `:normal`  - First signal at bottom (default)
    - `:reverse` - First signal at top.

# Returns
`::Plots.Plot`: Shaded wiggles on top of current plot object.

# Examples
```julia
using Plots, WaveletsExt

# Generate random signals
x = randn(16, 5)

# ----- Build wiggles -----
# Method 1
wiggle(x)

# Method 2
p = Plot()
wiggle(p, x)
```

Translated by Nicholas Hausch -- MATLAB file provided by Naoki Saito. The previous MATLAB
version contributors are Anthony K. Booer (SLB) and Bradley Marchand (NSWC-PC).  

Revised by Naoki Saito, Feb. 05, 2018. Maintained by Zeng Fung Liew for newest Julia version
compatibility. 

**See also:** [`wiggle!`](@ref)
"""
wiggle(args...; kwargs...) = wiggle!(Plots.Plot(), args...; kwargs...)
wiggle(plt::Plots.Plot, args...; kwargs...) = wiggle!(deepcopy(plt), args...; kwargs...)

"""
    wiggle!(wav[; taxis, zaxis, sc, EdgeColor, FaceColor, Orient, Overlap, ZDir])
    wiggle!(plt, wav[; taxis, zaxis, sc, EdgeColor, FaceColor, Orient, Overlap, ZDir])

Plot a set of shaded wiggles on the current displayed graphics or on top of `plt`. If there
are no displayed graphics currently available, a new `Plots.Plot` object is generated to
plot the shaded wiggles.

# Arguments
- `plt::Plots.Plot`: Input plot to plot shaded wiggles.
- `wav::AbstractArray{<:Number,2}`: Matrix of waveform columns.

# Keyword Arguments
- `taxis::AbstractVector`: (Default: `1:size(wav,1)`) Time axis vector
- `zaxis::AbstractVector`: (Default: `1:size(wav,2)`) Space axis vector
- `sc::Real`: (Default: `1`) Scale factor/magnification.
- `EdgeColor::Symbol`: (Default: `:black`) Sets edge of wiggles color.
- `FaceColor::Symbol`: (Default: `:black`) Sets shading color of wiggles.
- `Overlap::Bool`: (Default: `true`) How signals are scaled.
    - `true`  - Signals overlap (default);
    - `false` - Signals are scaled so they do not overlap.
- `Orient::Symbol`: (Default: `:across`) Controls orientation of wiggles.
    - `:across` - from left to right
    - `:down`   - from top to down
- `ZDir::Symbol`: (Default: `:normal`) Direction of space axis.
    - `:normal`  - First signal at bottom (default)
    - `:reverse` - First signal at top.

# Returns
`::Plots.Plot`: Shaded wiggles on top of current plot object.

# Examples
```julia
using Plots, WaveletsExt

# Generate random signals
x = randn(16, 5)

# ----- Build wiggles -----
# Build onto existing plot
plt = plot()
wiggle!(x)

# Build onto a specified plot
wiggle!(plt, x)
```

Translated by Nicholas Hausch -- MATLAB file provided by Naoki Saito. The previous MATLAB
version contributors are Anthony K. Booer (SLB) and Bradley Marchand (NSWC-PC).  

Revised by Naoki Saito, Feb. 05, 2018. Maintained by Zeng Fung Liew for newest Julia version
compatibility. 

**See also:** [`wiggle`](@ref)
"""
function wiggle!(args...; kwargs...)
    @nospecialize
    local plt
    try
        plt = current()
    catch
        return wiggle(args..., kwargs...)
    end
    wiggle!(current(), args...; kwargs...)
end

function wiggle!(plt::Plots.Plot,
                 wav::AbstractArray{T,2};
                 taxis::AbstractVector=1:size(wav,1),
                 zaxis::AbstractVector=1:size(wav,2),
                 sc::Real=1,
                 EdgeColor::Symbol=:black,
                 FaceColor::Symbol=:black,
                 Overlap::Bool=true,
                 Orient::Symbol=:across,
                 ZDir::Symbol=:normal) where T<:Number
    # Set axes
    (n, m) = size(wav)

    # Sanity check
    @assert Orient ∈ [:across, :down]
    @assert ZDir ∈ [:normal, :reverse]
    if length(taxis) != n
        error("Inconsistent taxis dimension!")
    end
    if length(zaxis) != m
        error("Inconsistent zaxis dimension!")
    end

    # For calculation purposes
    maxrow = zeros(m)
    minrow = zeros(m)
    for (i, wavᵢ) in enumerate(eachcol(wav))
        maxrow[i] = maximum(wavᵢ)
        minrow[i] = minimum(wavᵢ)
    end

    # Scale the data for plotting
    dz = mean(diff(zaxis))
    if Overlap
        wamp = 2 * dz * (sc/maximum(maxrow-minrow)) * wav
    else
        wmax = maximum(maxrow) <= 0 ? 0 : maximum(maxrow)
        wmin = minimum(minrow) >= 0 ? 0 : minimum(minrow)
        wamp = sc*wav/(wmax-wmin)
    end

    # Set initial plot
    t0 = minimum(taxis)
    t1 = maximum(taxis)
    z0 = minimum(zaxis)
    z1 = maximum(zaxis)
    if Orient == :down
        plot!(plt, xlims=(z0-dz,z1+dz), ylims=(t0,t1), yflip=true, legend=:none)
    else
        plot!(plt, xlims=(t0,t1), ylims=(z0-dz,z1+dz), legend=:none)
    end
    if ZDir == :reverse
        wamp = reverse(wamp, dims=2)
    end

    # Plot each wavelet
    for (i, wampᵢ) in enumerate(eachcol(wamp))
        t = deepcopy(taxis)
        w_sign = sign.(wampᵢ)
        for j in 1:(n-1)
            if (w_sign[j]!=w_sign[j+1] && w_sign[j]!=0 && w_sign[j+1]!=0)
                wampᵢ = [wampᵢ; 0]
                t = [t; t[j]-wampᵢ[j]*(t[j+1]-t[j])/(wampᵢ[j+1]-wampᵢ[j])]
            end
        end
        ix = sortperm(t)
        t = t[ix]
        wampᵢ = wampᵢ[ix]
        len = length(t)
        indperm = [1:len; len:-1:1]
        inputx = t[indperm]
        inputy = zaxis[i] .+ [wampᵢ; min.(wampᵢ[len:-1:1],0)]
        # In the plot! functions below, theoretically speaking, either fillrange = zaxis[k]
        # or fillrange=[zaxis[k], zaxis[k]+dz] should be used. However, those do not
        # generate the desired plots as of O3/19/2018. Somehow, the relative value of 0,
        # i.e., fillrange=0, works well, which is used temporarily.
        if Orient == :down
            plot!(plt, inputy, inputx, fillrange=0, fillalpha=0.75, 
                  fillcolor=FaceColor, linecolor=EdgeColor, orientation=:v)
        else
            plot!(plt, inputx, inputy, fillrange=0, fillalpha=0.75,
                  fillcolor=FaceColor, linecolor=EdgeColor)
        end
    end
    return plt
end

end # end module