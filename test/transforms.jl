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
    
end
