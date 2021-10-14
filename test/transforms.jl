@testset "DWT" begin
    x = randn(8)
    wt = wavelet(WT.haar)

    # 1D transforms
    y1 = wpt(x, wt, 1)
    y2 = wpt(x, wt, 2)
    y3 = wpt(x, wt, 3)
    @test wpd(x, wt) == [x y1 y2 y3]
    
    y = Array{Float64,2}(undef, (8,4))
    wpd!(y, x, wt)
    @test y == [x y1 y2 y3]

    # 2D transforms
    x = randn(8,8)
    y = dwt(x, wt, 1)
    z1 = wpt(x, wt, 1)
    z2 = wpt(x, wt, 2)
    z3 = wpt(x, wt, 3)
    @test y == z1
    @test wpd(x, wt) == cat(x, z1, z2, z3, dims=3)
    @test dwt(x, wt) == wpt(x, wt, makequadtree(x, 3, :dwt))
end

@testset "SWT" begin
    x = randn(8)
    wt = wavelet(WT.db4)
    g, h = WT.makereverseqmfpair(wt)
    tree = maketree(x, :dwt)
    sm = 3
    @test isdwt(sdwt(x, wt, 3), wt) ≈ x
    @test isdwt(sdwt(x, wt), wt, sm) ≈ x
    @test swpt(x, wt) == swpd(x, wt)[:,8:15]
    @test swpt(x, wt, 3) == swpd(x, wt)[:,8:15]
    @test iswpt(swpt(x, wt), wt) ≈ x
    @test iswpt(swpt(x, wt), wt, sm) ≈ x
    @test iswpd(swpd(x, wt), wt) ≈ x
    @test iswpd(swpd(x, wt), wt, 2) ≈ x
    @test iswpd(swpd(x, wt), wt, tree) ≈ x
    @test iswpd(swpd(x, wt), wt, tree, sm) ≈ x
    @test iswpd(swpd(x, wt), wt, 2, sm) ≈ x
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

@testset "ACWT" begin
    # acwt (1D)
    x = randn(8)
    wt = wavelet(WT.db4)
    tree = maketree(x, :dwt)
    @test iacdwt(acdwt(x, wt)) ≈ x
    @test iacdwt(acdwt(x, wt, 2), wt) ≈ x
    @test acwpt(x, wt) == acwpd(x, wt)[:,8:15]
    @test acwpt(x, wt, 2) == acwpd(x, wt)[:,4:7]
    @test iacwpt(acwpt(x, wt)) ≈ x
    @test iacwpd(acwpd(x, wt)) ≈ x
    @test iacwpd(acwpd(x, wt), 2) ≈ x
    @test iacwpd(acwpd(x, wt), wt, 2) ≈ x
    @test iacwpd(acwpd(x, wt), tree) ≈ x
    @test iacwpd(acwpd(x, wt), wt, tree) ≈ x

    # acwt (2D)
    x₂ = randn(8,8)
    y₂ = acdwt(x₂, wavelet(WT.haar))
    @test iacdwt(y₂) ≈ x₂
end

@testset "Transform All" begin
    x = randn(8)
    xₙ = [x x x]
    wt = wavelet(WT.db4)
    
    # dwt
    y = dwt(x, wt)
    yₙ = [y y y]
    @test dwtall(xₙ, wt) == yₙ
    @test idwtall(yₙ, wt) ≈ xₙ

    # wpt
    y = wpt(x, wt)
    yₙ = [y y y]
    @test wptall(xₙ, wt) == yₙ
    @test iwptall(yₙ, wt) ≈ xₙ

    # wpd
    y = [x wpt(x,wt,1) wpt(x,wt,2) wpt(x,wt,3)]
    yₙ = cat(y,y,y, dims=3)
    @test wpdall(xₙ, wt, 3) ≈ yₙ
    @test iwpdall(yₙ, wt, 3) ≈ xₙ

    # acdwt
    y = acdwt(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test acdwtall(xₙ, wt) == yₙ
    @test iacdwtall(yₙ, wt) ≈ xₙ

    # acwpt
    y = acwpt(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test acwptall(xₙ, wt) == yₙ
    @test iacwptall(yₙ, wt) ≈ xₙ

    # acwpd
    y = acwpd(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test acwpdall(xₙ, wt, 3) ≈ yₙ
    @test iacwpdall(yₙ, wt, 3) ≈ xₙ

    # sdwt
    y = sdwt(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test sdwtall(xₙ, wt) == yₙ
    @test isdwtall(yₙ, wt) ≈ xₙ

    # swpt
    y = swpt(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test swptall(xₙ, wt) == yₙ
    @test iswptall(yₙ, wt) ≈ xₙ

    # swpd
    y = swpd(x, wt)
    yₙ = cat(y,y,y, dims=3)
    @test swpdall(xₙ, wt, 3) ≈ yₙ
    @test iswpdall(yₙ, wt, 3) ≈ xₙ
    @test iswpdall(yₙ, wt, 3, 4) ≈ xₙ
end