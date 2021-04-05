function generatedata(N::Integer, noise::Real)
    n = 128

    d1=DiscreteUniform(16,32); d2=DiscreteUniform(32,96);
    
    # Making cylinder signals
    cylinder=zeros(n,N);
    a=rand(d1,N); b=a+rand(d2,N);
    η=randn(N);
    for k=1:N
        cylinder[a[k]:b[k],k]=(6+η[k])*ones(b[k]-a[k]+1);
    end
    cylinder += noise*randn(n,N);     # adding noise
    
    # Making bell signals
    bell=zeros(n,N);
    a=rand(d1,N); b=a+rand(d2,N);
    η=randn(N);
    for k=1:N
        bell[a[k]:b[k],k]=(6+η[k])*collect(0:(b[k]-a[k]))/(b[k]-a[k]);
    end
    bell += noise*randn(n,N);         # adding noise
    
    # Making funnel signals
    funnel=zeros(n,N);
    a=rand(d1,N); b=a+rand(d2,N);
    η=randn(N);
    for k=1:N
        funnel[a[k]:b[k],k]=(6+η[k])*collect((b[k]-a[k]):-1:0)/(b[k]-a[k]);
    end
    funnel += noise*randn(n,N);       # adding noise
    return cylinder, bell, funnel
end

cylinder, bell, funnel = generatedata(100, 0.8)
X = hcat(cylinder, bell, funnel)
y = repeat(["cylinder","bell","funnel"], inner = 100)
wt = wavelet(WT.coif4)

@test length(ldb(X, y, wt)) == 5
@test length(ldb(X, y, wt, topk=10, m=10)) == 5
@test length(ldb(X, y, wt, dm=SymmetricRelativeEntropy())) == 5
@test length(ldb(X, y, wt, dm=LpEntropy())) == 5
@test length(ldb(
        X, y, wt, dm=HellingerDistance(), energy=ProbabilityDensity()
    )) == 5
@test length(ldb(X, y, wt, dp=FishersClassSeparability())) == 5
@test length(ldb(X, y, wt, dp=RobustFishersClassSeparability())) == 5