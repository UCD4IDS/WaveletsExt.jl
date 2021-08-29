# [Wavelet Transforms and Their Best Bases](@id transforms_manual)
As an extension to Wavelets.jl's wavelet packet transform and best basis functions `wpt` and `bestbasistree`, WaveletsExt goes one step further and brings a full decomposition function `wpd` and redundant transforms in the form of Stationary Wavelet Transform, Autocorrelation Wavelet Transform, and Shift-Invariant Wavelet Transform. Additionally, more advanced best basis algorithms for a group of signals such as the Joint Best Basis (JBB) and Least Statistically Dependent Basis (LSDB) are also included here.

## Regular Wavelet Packet Transform

The standard best basis algorithm on the wavelet packet transform from Wavelets.jl can be performed as follows:
```@example wt
using Wavelets, WaveletsExt

# define function and wavelet
x = generatesignals(:heavysine, 8)
wt = wavelet(WT.db4)

# best basis tree
tree = bestbasistree(x, wt)

# decomposition
y = wpt(x, wt, tree); 
nothing # hide
```

However, in the case where there is a large amount of signals to transform to its best basis, one may find the following approach more convenient.
```@example wt
# generate 10 Heavy Sine signals
X = duplicatesignals(x, 10, 2, true, 0.5)

# decomposition of all signals
xw = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis trees, each column corresponds to 1 tree
trees = bestbasistree(xw, BB()); 
nothing # hide
```

Additionally, one can view the selected nodes from the best basis trees using the [`plot_tfbdry`](@ref WaveletsExt.Visualizations.plot_tfbdry) function as shown below.
```@example wt
# plot_tfbdry(trees[:,1])
```

We can also view the JBB and LSDB trees using a similar syntax. Unlike the previous best basis algorithm, JBB and LSDB do not generate a tree for each individual signal, as they search for the best tree that generalizes the group of signal as a whole.

* Joint Best Basis (JBB)
```@example wt
# joint best basis
tree = bestbasistree(xw, JBB())
# plot_tfbdry(tree)
```

* Least Statistically Dependent Basis (LSDB)
```@example wt
# least statistically dependent basis
tree = bestbasistree(xw, LSDB())
# plot_tfbdry(tree)
```

## Stationary and Autocorrelation Wavelet Transforms
The [Stationary Wavelet Transform (SWT)](https://link.springer.com/chapter/10.1007/978-1-4612-2544-7_17) was developed by G.P. Nason and B.W. Silverman in the 1990s. One can use the discrete SWT as shown below, or the SWT decomposition shown after.
### Example
```@example wt
# discrete swt
y = sdwt(x, wt)

# view the transform
# wiggle(y, sc=0.7)
```

The SWT decomposition and its best basis search is highly similar with that of the regular wavelet transforms.
* Regular best basis algorithm
```@example wt
# decomposition of all signals
xw = cat([swpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis trees, each column corresponds to 1 tree
trees = bestbasistree(xw, BB(redundant=true)); 
# plot_tfbdry(trees[:,1])
```

* Joint Best Basis (JBB)
```@example wt
# best basis trees, each column corresponds to 1 tree
tree = bestbasistree(xw, JBB(redundant=true)); 
# plot_tfbdry(tree)
```

* Least Statistically Dependent Basis (LSDB)
```@example wt
# best basis trees, each column corresponds to 1 tree
tree = bestbasistree(xw, LSDB(redundant=true)); 
# plot_tfbdry(tree)
```

In fact, this same exact procedures can be implemented with the [Autocorrelation wavelet transforms](https://www.math.ucdavis.edu/~saito/publications/saito_acs_spie.pdf), since they're both redundant types of transform. To implement the autocorrelation transforms, simply change `sdwt` to `acwt`, and `swpd` to `acwpt`.

## Shift-Invariant Wavelet Packet Decomposition
The [Shift-Invariant Wavelet Decomposition (SIWPD)](https://israelcohen.com/wp-content/uploads/2018/05/ICASSP95.pdf) is developed by I. Cohen. While it is also a type of redundant transform, it does not follow the same methodology as the SWT and the ACWT. One can compute the SIWPD of a single signal as follows.
### Example
```@example wt
# decomposition
xw = siwpd(x, wt)

# best basis tree
tree = bestbasistree(xw, 8, SIBB());
nothing # hide
```

As of right now, there is not too many functions written based on the SIWPD, as it does not follow the conventional style of wavelet transforms. There is a lot of ongoing work to develop more functions catered for the SIWPD such as it's inverse transforms and group-implementations.