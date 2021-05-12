X, y = generateclassdata(ClassData(:tri, 5, 5, 5))
wt = wavelet(WT.haar)

# AsymmetricRelativeEntropy + TimeFrequency + BasisDiscriminantMeasure
f = LocalDiscriminantBasis(wt, max_dec_level=4, top_k=5, n_features=5)
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
f = LocalDiscriminantBasis(wt, max_dec_level=4, dm=SymmetricRelativeEntropy(), 
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

# LpEntropy + TimeFrequency + RobustFishersClassSeparability
f = LocalDiscriminantBasis(wt, max_dec_level=4, dm=LpEntropy(), 
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
f = LocalDiscriminantBasis(wt, max_dec_level=4, dm=HellingerDistance(), 
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

# change number of features
@test_nowarn change_nfeatures(f, Xc, 5)
x = change_nfeatures(f, Xc, 5)
@test size(x) == (5, 15)
@test_logs (:warn, "Proposed n_features larger than currently saved n_features. Results will be less accurate since inverse_transform and transform is involved.") change_nfeatures(f, Xc, 10)
@test_throws ArgumentError change_nfeatures(f, Xc, 10)
