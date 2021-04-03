module Visualizations
export 
    plot_tfbdry,
    wiggle,
    wiggle!

using
    Plots
using
    ..Utils

"""
    selectednodes_matrix(x) 

Given a binary tree, output the matrix representation of the binary tree.
"""
function selectednodes_matrix(x::BitVector)  
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
    plot_tfbdry(tree[; start=0, nodecolor:white])

Given a tree, output a visual representation of the leaf nodes, user will 
have the option to start the node count of each level with 0 or 1.
"""
function plot_tfbdry(tree::BitVector; start::Integer=0, 
        nodecolor::Symbol=:white)

    @assert 0 <= start <= 1
    leaf = getleaf(tree)

    ncol = (length(leaf) + 1) >> 1
    nrow = Int(log2(ncol)) + 1
    mat = selectednodes_matrix(leaf)

    p = heatmap(start:(ncol+start-1), 0:(nrow-1), mat, 
        color = [:black, nodecolor], legend = false, 
        background_color = :black)

    plot!(p, xlims = (start-0.5, ncol+start-0.5), 
        ylims = (-0.5, nrow-0.5), yticks = 0:nrow)
    # plot horizontal lines
    for j in 0:(nrow-1)
        plot!(p, (start-0.5):(ncol+start-0.5), (j-0.5)*ones(ncol+1), 
            color = :white, legend = false)
    end

    # plot vertical lines
    for j in 1:(nrow-1)
        for jj in 1:2^(j-1)
            vpos = (ncol/2^j)*(2*jj-1)+start-0.5;
            plot!(p, vpos*ones(nrow-j+1), j-0.5:nrow-0.5, color = :white)
        end
    end
    plot!(p, (ncol+start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = :white)
    plot!(p, (start-0.5)*ones(nrow+1), -0.5:nrow-0.5, color = :white)
    plot!(p, yaxis = :flip)

    return p
end

"""
    wiggle(wav; taxis=1:size(wav,1), zaxis=1:size(wav,2), sc=1, 
        EdgeColor=:black, FaceColor=:black, Orient=:across, Overlap=true, 
        ZDir=:normal)

Plot a set of shaded wiggles.

# Arguments
- 'wav::AbstractArray{<:Number, 2}': matrix of waveform columns.
- 'taxis::Vector{<:Number}=1:size(wav,1)': time axis vector
- 'zaxis::Vector{<:Number}=1:size(wav,2)': space axis vector
- 'sc::AbstractFloat=1': scale factor/magnification.
- 'EdgeColor::Symbol=:black': Sets edge of wiggles color.
- 'FaceColor::Symbol=:black': Sets shading color of wiggles.
- 'Overlap::Bool=true': How signals are scaled.
        true  - Signals overlap (default);
        false - Signals are scaled so they do not overlap.
- 'Orient::Symbol=:across': Controls orientation of wiggles.
        :across - from left to right
        :down   - from top to down
- 'ZDir::Symbol=:normal': Direction of space axis.
        :normal  - First signal at bottom (default)
        :reverse - First signal at top.

Translated by Nicholas Hausch -- MATLAB file provided by Naoki Saito
The previous MATLAB version contributors are:
    Anthony K. Booer (SLB) and Bradley Marchand (NSWC-PC)
Revised by Naoki Saito, Feb. 05, 2018
"""
function wiggle(wav::AbstractArray{T,2}; taxis::Vector{S}=1:size(wav,1), 
        zaxis::Vector{S}=1:size(wav,2), sc::AbstractFloat=1, 
        EdgeColor::Symbol=:black, FaceColor::Symbol=:black, 
        Overlap::Bool=true, Orient::Symbol=:across, ZDir::Symbol=:normal) where
        {T<:Number, S<:Number}
    
    # Set axes
    (n,m) = size(wav)
    
    # Sanity check
    if length(taxis) != n
        error("Inconsistent taxis dimension!")
    end
    if length(zaxis) != m
        error("Inconsistent zaxis dimension!")
    end
    
    # For calculation purposes
    maxrow = zeros(m); minrow = zeros(m)
    for k = 1:m
      maxrow[k] = maximum(wav[:,k]); minrow[k] = minimum(wav[:,k])
    end
    
    # Scale the data for plotting
    wamp = deepcopy(wav)
    dt = mean(diff(taxis))
    dz = mean(diff(zaxis))
        if Overlap
      wamp *= 2 * dz * (sc/maximum(maxrow-minrow))
    else
        wmax = maximum(maxrow); wmin = minimum(minrow);
        if wmax<=0
            wmax = 0
      end
        if wmin>=0
            wmin = 0
      end
          wamp = sc*wav/(wmax-wmin)
    end
    
    # Set initial plot
    t0 = minimum(taxis)
    t1 = maximum(taxis)
    z0 = minimum(zaxis)
    z1 = maximum(zaxis)
    if Orient == :down
    plot(xlims=(z0-dz,z1+dz), ylims=(t0,t1), yflip=true, legend=:none)
    else
    plot(xlims=(t0,t1), ylims=(z0-dz,z1+dz), legend=:none)
    end
    if ZDir == :reverse
        wamp = flipdim(wamp,2)
    end
    
    # Plot each wavelet
    for k = 1:m
        sig = wamp[:,k]
        t = deepcopy(taxis)
        w_sign = sign.(sig)
        for j=1:n-1
            if (w_sign[j]!=w_sign[j+1] && w_sign[j]!=0 && w_sign[j+1]!=0)
            sig = [sig; 0]
            t = [t; t[j]-sig[j]*(t[j+1]-t[j])/(sig[j+1]-sig[j])]
            end
        end
        IX = sortperm(t)
        t = t[IX]
        sig = sig[IX]
        len = length(t)
        len1 = collect(len:-1:1)
        indperm = [1:len;len1]
        inputx = t[indperm]
        inputy = zaxis[k] .+ [sig;min.(sig[len1],0)]
        # In the plot! functions below, theoretically speaking, either
        # fillrange = zaxis[k] or fillrange=[zaxis[k], zaxis[k]+dz] should be 
        # used. However, those do not generate the desired plots as of 
        # O3/19/2018. Somehow, the relative value of 0, i.e., fillrange=0, works 
        # well, which is used temporarily.
        if Orient == :down
            plot!(
                inputy, inputx, fillrange=0, fillalpha=0.75, 
                fillcolor=FaceColor, linecolor=EdgeColor, orientation=:v
            )
        else
            plot!(
                inputx, inputy, fillrange=0, fillalpha=0.75, 
                fillcolor=FaceColor, linecolor=EdgeColor
            )
        end
    end
    plot!() # flushing the display.
end

"""
    wiggle!(wav; taxis=1:size(wav,1), zaxis=1:size(wav,2), sc=1, 
        EdgeColor=:black, FaceColor=:black, Orient=:across, Overlap=true, 
        ZDir=:normal)

Plot a set of shaded wiggles on the current displayed graphics

# Arguments
- 'wav::AbstractArray{<:Number,2}': matrix of waveform columns.
- 'taxis::Vector{<:Number}=1:size(wav,1)': time axis vector
- 'zaxis::Vector{<:Number}=1:size(wav,2)': space axis vector
- 'sc::AbstractFloat=1': scale factor/magnification.
- 'EdgeColor::Symbol=:black': Sets edge of wiggles color.
- 'FaceColor::Symbol=:black': Sets shading color of wiggles.
- 'Overlap::Bool=true': How signals are scaled.
        true  - Signals overlap (default);
        false - Signals are scaled so they do not overlap.
- 'Orient::Symbol=:across': Controls orientation of wiggles.
        :across - from left to right
        :down   - from top to down
- 'ZDir::Symbol=:normal': Direction of space axis.
        :normal  - First signal at bottom (default)
        :reverse - First signal at top.

Translated by Nicholas Hausch -- MATLAB file provided by Naoki Saito
The previous MATLAB version contributors are:
    Anthony K. Booer (SLB) and Bradley Marchand (NSWC-PC)
Revised by Naoki Saito, Feb. 05, 2018
"""
function wiggle!(wav::AbstractArray{T,2}; taxis::Vector{S}=1:size(wav,1), 
        zaxis::Vector{S}=1:size(wav,2), sc::AbstractFloat=1, 
        EdgeColor::Symbol=:black, FaceColor::Symbol=:black, 
        Overlap::Bool=true, Orient::Symbol=:across, ZDir::Symbol=:normal) where
        {T<:Number, S<:Number}

    # Set axes
    (n,m) = size(wav)
    
    # Sanity check
    if length(taxis) != n
        error("Inconsistent taxis dimension!")
    end
    if length(zaxis) != m
        error("Inconsistent zaxis dimension!")
    end
    
    # For calculation purposes
    maxrow = zeros(m); minrow = zeros(m)
    for k = 1:m
        maxrow[k] = maximum(wav[:,k]); minrow[k] = minimum(wav[:,k])
    end
    
    # Scale the data for plotting
    wamp = deepcopy(wav)
    dt = mean(diff(taxis))
    dz = mean(diff(zaxis))
    if Overlap
        wamp *= 2 * dz * (sc/maximum(maxrow-minrow))
    else
        wmax = maximum(maxrow); wmin = minimum(minrow);
        if wmax<=0
            wmax = 0
        end
        if wmin>=0
            wmin = 0
        end
            wamp = sc*wav/(wmax-wmin)
    end
    
    # Set initial plot
    t0 = minimum(taxis)
    t1 = maximum(taxis)
    z0 = minimum(zaxis)
    z1 = maximum(zaxis)
    if Orient == :down
        plot!(xlims=(z0-dz,z1+dz), ylims=(t0,t1), yflip=true, legend=:none)
    else
        plot!(xlims=(t0,t1), ylims=(z0-dz,z1+dz), legend=:none)
    end
    if ZDir == :reverse
        wamp = flipdim(wamp,2)
    end
    
    # Plot each wavelet
    for k = 1:m
        sig = wamp[:,k]
        t = deepcopy(taxis)
        w_sign = sign.(sig)
        for j=1:n-1
            if (w_sign[j]!=w_sign[j+1] && w_sign[j]!=0 && w_sign[j+1]!=0)
                sig = [sig; 0]
                t = [t; t[j]-sig[j]*(t[j+1]-t[j])/(sig[j+1]-sig[j])]
            end
        end
        IX = sortperm(t)
        t = t[IX]
        sig = sig[IX]
        len = length(t)
        len1 = collect(len:-1:1)
        indperm = [1:len;len1]
        inputx = t[indperm]
        inputy = zaxis[k] .+ [sig;min.(sig[len1],0)]
        # In the plot! functions below, theoretically speaking, either
        # fillrange = zaxis[k] or fillrange=[zaxis[k], zaxis[k]+dz] should be 
        # used. However, those do not generate the desired plots as of 
        # O3/19/2018. Somehow, the relative value of 0, i.e., fillrange=0, works 
        # well, which is used temporarily.
        if Orient == :down
            plot!(
                inputy, inputx, fillrange=0, fillalpha=0.75, 
                fillcolor=FaceColor, linecolor=EdgeColor, orientation=:v
            )
        else
            plot!(
                inputx, inputy, fillrange=0, fillalpha=0.75, 
                fillcolor=FaceColor, linecolor=EdgeColor
            )
        end
    end
    plot!() # flushing the display.
end

end # end module