@testset "WPD" begin
    x = randn(8)
    wt = wavelet(WT.haar)
    g, h = WT.makereverseqmfpair(wt, true)
    y1 = wpt(x, wt, 1)
    y2 = wpt(x, wt, 2)
    y3 = wpt(x, wt, 3)
    @test wpd(x, wt) == [x y1 y2 y3]
    @test wpd(x, wt, h, g) == [x y1 y2 y3]
    
    y = Array{Float64,2}(undef, (8,4))
    wpd!(y, x, wt)
    @test y == [x y1 y2 y3]
    wpd!(y, x, wt, h, g)
    @test y == [x y1 y2 y3]
end

@testset "SWT" begin
    x = randn(8)
    wt = wavelet(WT.haar)
    tree = maketree(x, :dwt)
    ε = [true, true, false]
    @test isdwt(sdwt(x, wt, 3), wt) ≈ x
    @test isdwt(sdwt(x, wt), wt, ε) ≈ x
    @test swpt(x, wt) == swpd(x, wt)[:,8:15]
    @test swpt(x, wt, 3) == swpd(x, wt)[:,8:15]
    @test swpt(x, wt, tree) == sdwt(x, wt)
    @test iswpt(swpd(x, wt), wt, tree) ≈ x
    @test iswpt(swpd(x, wt), wt, ε, tree) ≈ x
    @test iswpt(swpd(x, wt), wt, ε) ≈ x
end

@testset "SIWPD" begin
    # siwpd 
    x = randn(4)
    wt = wavelet(WT.haar)
    y = siwpd(x, wt, 2, 1)
    y0 = y[1,4:7]
    y1 = y[2,4:7]
    y2 = y[3,4:7]
    y3 = y[4,4:7]
    @test y0 == wpt(x, wt)
    @test !all([isdefined(y1, i) for i in 1:4])
    @test y2 == wpt(circshift(x,2), wt)
    @test !all([isdefined(y3, i) for i in 1:4])
    # make tree
    tree = [
        trues(1), 
        repeat([trues(2)],2)..., 
        repeat([BitVector([1,0,1,0])],4)...
    ]
    @test makesiwpdtree(4, 2, 1) == tree
end
