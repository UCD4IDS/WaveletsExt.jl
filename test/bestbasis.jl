x = randn(16,5)         # Set of 1D signals
y = randn(16,16,5)      # Set of 2D signals
wt = wavelet(WT.haar)

xw = wpdall(x,wt)
xsw = swpdall(x,wt)
xacw = acwpdall(x,wt)
yw = wpdall(y,wt)
ysw = swpdall(y,wt)
# yacw = acwpdall(y,wt)     # TODO: not available yet

# bb
@test isvalidtree(x[:,1], bestbasistree(xw[:,:,1], BB()))
@test isvalidtree(x[:,1], bestbasistreeall(xw, BB())[:,1])
@test isvalidtree(x[:,1], bestbasistreeall(xw, BB(LogEnergyEntropyCost(), false))[:,1])                          
@test isvalidtree(x[:,1], bestbasistreeall(xsw, BB(redundant=true))[:,1])
@test isvalidtree(x[:,1], bestbasistreeall(xacw, BB(redundant=true))[:,1])  
@test isvalidtree(y[:,:,1], bestbasistree(yw[:,:,:,1], BB()))
@test isvalidtree(y[:,:,1], bestbasistreeall(yw, BB())[:,1])
@test isvalidtree(y[:,:,1], bestbasistreeall(yw, BB(LogEnergyEntropyCost(), false))[:,1])                          
@test isvalidtree(y[:,:,1], bestbasistreeall(ysw, BB(redundant=true))[:,1])
# @test isvalidtree(y[:,:,1], bestbasistreeall(yacw, BB(redundant=true))[:,1])  

# jbb
@test isvalidtree(x[:,1], bestbasistree(xw))
@test isvalidtree(x[:,1], bestbasistree(xw, JBB(NormCost(), false)))
@test isvalidtree(x[:,1], bestbasistree(xsw, JBB(redundant=true)))
@test isvalidtree(x[:,1], bestbasistree(xacw, JBB(redundant=true)))
@test isvalidtree(y[:,:,1], bestbasistree(yw))
@test isvalidtree(y[:,:,1], bestbasistree(yw, JBB(NormCost(), false)))
@test isvalidtree(y[:,:,1], bestbasistree(ysw, JBB(redundant=true)))
# @test isvalidtree(y[:,:,1], bestbasistree(yacw, JBB(redundant=true)))

# lsdb
@test isvalidtree(x[:,1], bestbasistree(xw, LSDB()))
@test isvalidtree(x[:,1], bestbasistree(xsw, LSDB(redundant=true)))
@test isvalidtree(x[:,1], bestbasistree(xacw, LSDB(redundant=true)))
@test isvalidtree(y[:,:,1], bestbasistree(yw, LSDB()))
@test isvalidtree(y[:,:,1], bestbasistree(ysw, LSDB(redundant=true)))
# @test isvalidtree(y[:,:,1], bestbasistree(yacw, LSDB(redundant=true)))

# misc
@test_throws ArgumentError BestBasis.bestbasis_treeselection(randn(15), 8, :fail)
@test_throws AssertionError BestBasis.bestbasis_treeselection(randn(7), 3, :fail) # true n=4
