@testset "DNFT" begin
    @test typeof(VisuShrink(HardTH(), 8)) == VisuShrink
    @test typeof(RelErrorShrink()) == RelErrorShrink
    @test typeof(RelErrorShrink(HardTH())) == RelErrorShrink
    @test typeof(RelErrorShrink(HardTH(), 1)) == RelErrorShrink
    x = randn(8)
    tree = maketree(x, :full)
    @test typeof(SureShrink(HardTH(), 1)) == SureShrink
    @test typeof(SureShrink(x)) == SureShrink
    @test typeof(SureShrink(x, false, tree)) == SureShrink
    @test typeof(SureShrink(x, false, tree, HardTH())) == SureShrink
end

@testset "Denoise" begin
    # Single denoising
    n = 2^8
    x0 = generatesignals(:heavysine, 8)
    x = x0 + 0.5*randn(n)
    wt = wavelet(WT.haar)
    err = relativenorm(x, x0)
    dnt = VisuShrink(2, HardTH())

    ## Non-redundant Wavelet Transforms
    y = denoise(x, :sig, wt, dnt=dnt)
    @test relativenorm(y, x0) <= err
    y = denoise(dwt(x, wt, 4), :dwt, wt, L=4, dnt=dnt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err
    y = denoise(dwt(x, wt), :dwt, wt, dnt=dnt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err
    y = denoise(
        wpt(x, wt), :wpt, wt, tree=maketree(x, :full), dnt=dnt, 
        smooth=:undersmooth
    )
    @test relativenorm(y, x0) <= 2*err

    ## Stationary Wavelet Transforms
    y = denoise(sdwt(x, wt), :sdwt, wt, dnt=dnt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err
    y = denoise(swpd(x, wt), :swpd, wt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err

    ## Autocorrelation wavelet transforms
    y = denoise(acdwt(x, wt), :acwt, wt, dnt=dnt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err
    y = denoise(acwpd(x, wt), :acwpt, wt, smooth=:undersmooth)
    @test relativenorm(y, x0) <= 2*err
    
    # Group denoising
    x0 = duplicatesignals(generatesignals(:heavysine, 8), 5, 2)
    x = duplicatesignals(generatesignals(:heavysine, 8), 5, 2, true, 0.5)
    max_err = maximum([relativenorm(x[:,i],x0[:,i]) for i in 1:5])
    dnt = VisuShrink(2, HardTH())

    ## Non-redundant Transforms
    y = denoiseall(x, :sig, wt, dnt=dnt, bestTH=mean)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(dwtall(x, wt), :dwt, wt, dnt=dnt)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    dnt = RelErrorShrink(HardTH(), 0.3)
    y = denoiseall(wptall(x, wt), :wpt, wt, tree=maketree(n, 8, :full), dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(wptall(x, wt), :wpt, wt, tree=maketree(n, 8, :full), dnt=dnt, estnoise=relerrorthreshold, bestTH=mean)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err

    ## Stationary Wavelet Transforms
    y = denoiseall(cat([sdwt(x[:,i], wt) for i in 1:5]..., dims=3), :sdwt, wt)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(cat([sdwt(x[:,i], wt) for i in 1:5]..., dims=3), :sdwt, wt,
        dnt=dnt, estnoise=relerrorthreshold, bestTH=mean)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(cat([swpd(x[:,i], wt) for i in 1:5]..., dims=3), :swpd, wt,
        tree=maketree(n, 7, :full), dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
    y = denoiseall(cat([swpd(x[:,i], wt) for i in 1:5]..., dims=3), :swpd, wt,
        tree=maketree(n, 7, :full), dnt=dnt, estnoise=relerrorthreshold, 
        bestTH=mean)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err

    ## Autocorrelation Wavelet Transforms
    y = denoiseall(cat([acdwt(x[:,i], wt) for i in 1:5]..., dims=3), :acwt, wt)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err 
    y = denoiseall(cat([acwpd(x[:,i], wt) for i in 1:5]..., dims=3), :acwpt, wt,
        tree=maketree(n, 7, :full), dnt=dnt, estnoise=relerrorthreshold)
    @test mean([relativenorm(y[:,i],x0[:,i]) for i in 1:5]) <= max_err
end

@testset "Threshold Determination" begin
    # non-stationary
    x = randn(32)
    tree = maketree(32, 5, :full)
    @test typeof(noisest(x, false)) <: Real
    @test typeof(Denoising.surethreshold(x, false)) <: Real
    @test typeof(relerrorthreshold(x, false)) <: Real
    # stationary - sdwt
    x = randn(32, 6)
    @test typeof(noisest(x, true, nothing)) <: Real
    @test typeof(Denoising.surethreshold(x, true, nothing)) <: Real
    @test typeof(relerrorthreshold(x, true, nothing)) <: Real
    # stationary - swpd
    x = randn(32, 63)
    @test typeof(noisest(x, true, tree)) <: Real
    @test typeof(Denoising.surethreshold(x, true, tree)) <: Real
    @test typeof(relerrorthreshold(x, true, tree)) <: Real
    # relative error plot
    @test typeof(relerrorthreshold(x, true, tree, makeplot=true)[2]) == 
        Plots.Plot{Plots.GRBackend}
end