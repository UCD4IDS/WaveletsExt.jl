# Best Basis Examples

## Regular Best Basis 
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
X = generatesignals(x, 10, 2, true, 0.5)
wt = wavelet(WT.db4)

# decomposition
y = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(y, BB())
```

## Joint Best Basis (JBB)
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
X = generatesignals(x, 10, 2, true, 0.5)
wt = wavelet(WT.db4)

# decomposition
y = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(y, JBB())
```

## Least Statistically Dependent Basis (LSDB)
```@example
using Wavelets, WaveletsExt

# define function and wavelet
x = testfunction(256, "HeaviSine")
X = generatesignals(x, 10, 2, true, 0.5)
wt = wavelet(WT.db4)

# decomposition
y = cat([wpd(X[:,i], wt) for i in axes(X,2)]..., dims=3)

# best basis tree
bt = bestbasistree(y, LSDB())
```