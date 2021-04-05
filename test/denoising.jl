@testset "DNFT" begin
    @test typeof(VisuShrink(HardTH(), 8)) == VisuShrink
    @test typeof(RelErrorShrink()) == RelErrorShrink
    @test typeof(RelErrorShrink(HardTH())) == RelErrorShrink
    @test typeof(RelErrorShrink(HardTH(), 1)) == RelErrorShrink
    x = randn(8)
    tree = maketree(x, :full)
    @test typeof(SureShrink(HardTH(), 1)) == SureShrink
    @test typeof(SureShrink(x)) == SureShrink
    @test typeof(SureShrink(x, tree)) == SureShrink
    @test typeof(SureShrink(x, tree, HardTH())) == SureShrink
end

@testset "Denoise" begin
    # single denoising
    n = 2^8
    x0 = testfunction(n, "HeaviSine")
    x = x0 + 0.5*randn(n)
    wt = wavelet(WT.haar)
    err = relativenorm(x, x0)
    dnt = VisuShrink(HardTH(), 5)
    y = denoise(x, :sig, wt, dnt=dnt)
    @test relativenorm(y, x0) <= err
    y = denoise(dwt(x, wt, 4), :dwt, wt, L=4, dnt=dnt)
    @test relativenorm(y, x0) <= err
    y = denoise(dwt(x, wt), :dwt, wt, dnt=dnt)
    @test relativenorm(y, x0) <= err
    y = denoise(wpt(x, wt), :wpt, wt, tree=maketree(x, :full), dnt=dnt)
    @test relativenorm(y, x0) <= err
    dnt = RelErrorShrink(HardTH(), 0.3)
    y = denoise(sdwt(x, wt), :sdwt, wt, dnt=dnt, estnoise=relerrorthreshold)
    @test relativenorm(y, x0) <= err
    y = denoise(swpd(x, wt), :swpd, wt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= err
    # group denoising
    x0 = generatesignals(testfunction(n, "HeaviSine"), 5, 2)
    x = generatesignals(testfunction(n, "HeaviSine"), 5, 2, true, 0.5)
    max_err = maximum([relativenorm(x[:,i],x0[:,i]) for i in 1:5])
    dnt = VisuShrink(HardTH(), 5)
    y = denoiseall(x, :sig, wt, dnt=dnt, bestTH=mean)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(hcat([dwt(x[:,i], wt) for i in 1:5]...), :dwt, wt, dnt=dnt)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    dnt = RelErrorShrink(HardTH(), 0.3)
    y = denoiseall(hcat([wpt(x[:,i], wt) for i in 1:5]...), :wpt, wt,
        tree=maketree(n, 8, :full), dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(cat([sdwt(x[:,i], wt) for i in 1:5]..., dims=3), :sdwt, wt,
        dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(cat([swpd(x[:,i], wt) for i in 1:5]..., dims=3), :swpd, wt,
        tree=maketree(n, 7, :full), dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
end