@testset "DWT" begin
    # Single steps
    x = [2,3,-4,5.0]
    wt = wavelet(WT.db4)
    g, h = WT.makereverseqmfpair(wt, true)
    w₁, w₂ = DWT.dwt_step(x,h,g)
    w₁ = round.(w₁, digits=3); w₂ = round.(w₂, digits=3)
    @test [w₁;w₂] == [-0.524, 4.767, 1.803, 5.268]
    @test round.(DWT.idwt_step(w₁, w₂, h, g), digits=3) == x
    
    wt = wavelet(WT.db4)
    # 1D transforms
    x = randn(8)
    y1 = wpt(x, wt, 1)
    y2 = wpt(x, wt, 2)
    y3 = wpt(x, wt, 3)
    @test wpd(x, wt) ≈ [x y1 y2 y3]
    @test wpd!(zeros(8,4), x, wt) ≈ [x y1 y2 y3]
    @test iwpd(wpd(x,wt),wt) ≈ x
    @test iwpd(wpd(x,wt),wt,2) ≈ x
    @test iwpd(wpd(x,wt),wt,maketree(x,:dwt)) ≈ x

    # 2D transforms
    x = randn(8,8)
    y = dwt(x, wt, 1)
    z1 = wpt(x, wt, 1)
    z2 = wpt(x, wt, 2)
    z3 = wpt(x, wt, 3)
    @test y ≈ z1
    @test wpd(x, wt) ≈ cat(x, z1, z2, z3, dims=3)
    @test dwt(x, wt) ≈ wpt(x, wt, maketree(x,:dwt))
    @test iwpd(wpd(x,wt),wt) ≈ x
    @test iwpd(wpd(x,wt),wt,2) ≈ x
    @test iwpd(wpd(x,wt),wt,maketree(x,:dwt)) ≈ x
    @test iwpt(wpt(x,wt),wt) ≈ x
    @test iwpt(wpt(x,wt,2),wt,2) ≈ x
    @test iwpt(wpt(x,wt,maketree(x,:dwt)),wt,maketree(x,:dwt)) ≈ x
end

@testset "SWT" begin
    # Single step
    x = [2,3,-4,5.0]
    wt = wavelet(WT.db4)
    g, h = WT.makereverseqmfpair(wt, true)
    w₁, w₂ = SWT.sdwt_step(x, 0, h, g)
    w₁ = round.(w₁,digits=3); w₂ = round.(w₂,digits=3)
    @test [w₁ w₂] == [3.854 -6.181;-0.524 1.803;0.389 -0.89;4.767 5.268]
    @test round.(SWT.isdwt_step(w₁, w₂, 0, h, g), digits=3) == x

    # 1D transforms
    x = randn(8)
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

    # 2D transforms
    x = randn(8,8)
    tree = maketree(x, :dwt)
    sm = 3
    @test isdwt(sdwt(x, wt, 3), wt) ≈ x
    @test isdwt(sdwt(x, wt), wt, sm) ≈ x
    @test swpt(x, wt) == swpd(x, wt)[:,:,22:85]
    @test swpt(x, wt, 3) == swpd(x, wt)[:,:,22:85]
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
    x = randn(8)    # 1D
    xₙ = [x x x]
    w = randn(8,8)  # 2D
    wₙ = cat(w,w,w,dims=3)
    wt = wavelet(WT.db4)
    
    # dwt
    y = dwt(x, wt); z = dwt(w, wt)
    yₙ = [y y y]; zₙ = cat(z,z,z,dims=3)
    @test dwtall(xₙ, wt) == yₙ
    @test dwtall(wₙ, wt) == zₙ
    @test idwtall(yₙ, wt) ≈ xₙ
    @test idwtall(zₙ, wt) ≈ wₙ

    # wpt
    y = wpt(x, wt); z = wpt(w, wt)
    yₙ = [y y y]; zₙ = cat(z,z,z,dims=3)
    @test wptall(xₙ, wt) == yₙ
    @test wptall(wₙ, wt) == zₙ
    @test iwptall(yₙ, wt) ≈ xₙ
    @test iwptall(zₙ, wt) ≈ wₙ

    # wpd
    y = wpd(x,wt); z = wpd(w,wt)
    yₙ = cat(y,y,y, dims=3); zₙ = cat(z,z,z,dims=4)
    @test wpdall(xₙ, wt, 3) ≈ yₙ
    @test wpdall(wₙ, wt, 3) ≈ zₙ
    @test iwpdall(yₙ, wt, 3) ≈ xₙ
    @test iwpdall(zₙ, wt, 3) ≈ wₙ

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
    y = sdwt(x, wt); z = sdwt(w,wt)
    yₙ = cat(y,y,y, dims=3); zₙ = cat(z,z,z,dims=4)
    @test sdwtall(xₙ, wt) == yₙ
    @test sdwtall(wₙ, wt) == zₙ
    @test isdwtall(yₙ, wt) ≈ xₙ
    @test isdwtall(zₙ, wt) ≈ wₙ

    # swpt
    y = swpt(x, wt); z = swpt(w,wt)
    yₙ = cat(y,y,y, dims=3); zₙ = cat(z,z,z,dims=4)
    @test swptall(xₙ, wt) == yₙ
    @test swptall(wₙ, wt) == zₙ
    @test iswptall(yₙ, wt) ≈ xₙ
    @test iswptall(zₙ, wt) ≈ wₙ

    # swpd
    y = swpd(x, wt); z = swpd(w,wt)
    yₙ = cat(y,y,y, dims=3); zₙ = cat(z,z,z,dims=4)
    @test swpdall(xₙ, wt, 3) ≈ yₙ
    @test swpdall(wₙ, wt, 3) ≈ zₙ
    @test iswpdall(yₙ, wt, 3) ≈ xₙ
    @test iswpdall(zₙ, wt, 3) ≈ wₙ
    @test iswpdall(yₙ, wt, 3, 4) ≈ xₙ
    @test iswpdall(zₙ, wt, 3, 4) ≈ wₙ
end