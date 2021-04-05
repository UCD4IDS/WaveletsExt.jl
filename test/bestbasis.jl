x = randn(16,5)
wt = wavelet(WT.haar)
xw = cat([wpd(x[:,i], wt) for i in axes(x,2)]..., dims=3)
xsw = cat([swpd(x[:,i], wt) for i in axes(x,2)]..., dims=3)

# bb
@test isvalidtree(x[:,1], bestbasistree(xw, method=BB())[:,1])
@test isvalidtree(x[:,1], 
    bestbasistree(xw, BB(LogEnergyEntropyCost(), false))[:,1])                          
@test isvalidtree(x[:,1], bestbasistree(xsw, BB(stationary=true))[:,1])  

# jbb
@test isvalidtree(x[:,1], bestbasistree(xw))
@test isvalidtree(x[:,1], bestbasistree(xw, JBB(NormCost(), false)))
@test isvalidtree(x[:,1], bestbasistree(xsw, JBB(stationary=true)))

# lsdb
@test isvalidtree(x[:,1], bestbasistree(xw, method=LSDB()))
@test isvalidtree(x[:,1], bestbasistree(xw, LSDB()))

# siwpd_bb
x = randn(16)
xw0 = siwpd(x, wt)
xw4 = siwpd(circshift(x,4), wt)
bt0 = map(node -> sum(node), bestbasistree(xw0, 4, SIBB()))
bt4 = map(node -> sum(node), bestbasistree(xw4, 4, SIBB()))
@test bt0 == bt4

# get coefficients
tr = maketree(x)
xw = wpd(x, wt)
@test bestbasiscoef(xw, tr) == wpt(x, wt)
x = randn(16,5)
xw = cat([wpd(x[:,i], wt) for i in axes(x,2)]..., dims=3)
@test bestbasiscoef(xw, tr) == hcat([wpt(x[:,i], wt) for i in axes(x,2)]...)
@test bestbasiscoef(x, wt, tr) == hcat([wpt(x[:,i], wt) for i in axes(x,2)]...)
tr = BitArray(ones(15,5))
@test bestbasiscoef(xw, tr) == hcat([wpt(x[:,i], wt) for i in axes(x,2)]...)
@test bestbasiscoef(x, wt, tr) == hcat([wpt(x[:,i], wt) for i in axes(x,2)]...)
