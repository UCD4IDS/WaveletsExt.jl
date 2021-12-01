@testset "1D LDB" begin
    X, y = generateclassdata(ClassData(:tri, 5, 5, 5))
    wt = wavelet(WT.haar)
    
    # AsymmetricRelativeEntropy + TimeFrequency + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # SymmetricRelativeEntropy + TimeFrequency + FishersClassSeparability
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, dm=SymmetricRelativeEntropy(), 
        dp=FishersClassSeparability(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # LpDistance + TimeFrequency + RobustFishersClassSeparability
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, dm=LpDistance(), 
        dp=RobustFishersClassSeparability(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # HellingerDistance + ProbabilityDensity + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, dm=HellingerDistance(), 
        en=ProbabilityDensity(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # EarthMoverDistance + Signatures(equal weight) + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, dm=EarthMoverDistance(), 
        en=Signatures(:equal), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # EarthMoverDistance + Signatures(pdf weight) + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(wt=wt, max_dec_level=4, dm=EarthMoverDistance(), 
        en=Signatures(:pdf), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (32, 15) 
    
    # change number of features
    @test_nowarn change_nfeatures(f, Xc, 5)
    x = change_nfeatures(f, Xc, 5)
    @test size(x) == (5, 15)
    @test_logs (:warn, "Proposed n_features larger than currently saved n_features. Results will be less accurate since inverse_transform and transform is involved.") change_nfeatures(f, Xc, 10)
    @test_throws ArgumentError change_nfeatures(f, Xc, 10)
end

@testset "2D LDB" begin
    X = cat(rand(Normal(0,1), 8,8,5), rand(Normal(1,1), 8,8,5), rand(Normal(2,1), 8,8,5), dims=3)
    y = [repeat([1], 5); repeat([2],5); repeat([3],5)]
    
    # AsymmetricRelativeEntropy + TimeFrequency + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(max_dec_level=2, top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # SymmetricRelativeEntropy + TimeFrequency + FishersClassSeparability
    f = LocalDiscriminantBasis(max_dec_level=2, dm=SymmetricRelativeEntropy(), 
        dp=FishersClassSeparability(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # LpDistance + TimeFrequency + RobustFishersClassSeparability
    f = LocalDiscriminantBasis(max_dec_level=2, dm=LpDistance(), 
        dp=RobustFishersClassSeparability(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # HellingerDistance + ProbabilityDensity + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(max_dec_level=2, dm=HellingerDistance(), 
        en=ProbabilityDensity(), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # EarthMoverDistance + Signatures(equal weight) + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(max_dec_level=2, dm=EarthMoverDistance(), 
        en=Signatures(:equal), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # EarthMoverDistance + Signatures(pdf weight) + BasisDiscriminantMeasure
    f = LocalDiscriminantBasis(max_dec_level=2, dm=EarthMoverDistance(), 
        en=Signatures(:pdf), top_k=5, n_features=5)
    @test typeof(f) == LocalDiscriminantBasis
    @test_nowarn fit_transform(f, X, y)
    @test_nowarn fit!(f, X, y)
    @test_nowarn transform(f, X)
    Xc = transform(f, X)
    @test size(Xc) == (5,15)
    @test_nowarn inverse_transform(f, Xc)
    X̂ = inverse_transform(f, Xc)
    @test size(X̂) == (8, 8, 15) 
    
    # change number of features
    @test_nowarn change_nfeatures(f, Xc, 5)
    x = change_nfeatures(f, Xc, 5)
    @test size(x) == (5, 15)
    @test_logs (:warn, "Proposed n_features larger than currently saved n_features. Results will be less accurate since inverse_transform and transform is involved.") change_nfeatures(f, Xc, 10)
    @test_throws ArgumentError change_nfeatures(f, Xc, 10)
end