@testset "Indexing" begin
    @test maxtransformlevels(randn(4,2), 1) == 2
    @test maxtransformlevels(randn(4,2), 2) == 1
    @test_throws AssertionError maxtransformlevels(randn(4,2), 3)

    Xw = collect(1:12) |> x -> reshape(x,(4,3))
    @test getbasiscoef(Xw, maketree(4,2,:dwt)) == [9,10,7,8]
    @test getbasiscoef(Xw, maketree(4,2,:full)) == [9,10,11,12]
    @test_throws AssertionError getbasiscoef(Xw, BitVector([0,1,0]))
    @test_throws AssertionError getbasiscoef(Xw, BitVector([1,1,1,1]))
    @test_throws AssertionError getbasiscoef(Xw, maketree(8,2,:dwt))
    @test_throws AssertionError getbasiscoef(zeros(4,4), maketree(4,2,:dwt))
    @test_throws ArgumentError getbasiscoef(zeros(4,2), maketree(4,2,:dwt))

    Xw = collect(1:24) |> x -> reshape(x,(4,3,2))
    tree₁ = maketree(4,2,:dwt)
    tree₂ = maketree(4,2,:full)
    @test getbasiscoefall(Xw, tree₁) == [9 21;10 22;7 19;8 20]
    @test getbasiscoefall(Xw, tree₂) == [9 21;10 22;11 23;12 24]
    @test getbasiscoefall(Xw, [tree₁ tree₂]) == [9 21;10 22;7 23;8 24]
    @test_throws AssertionError getbasiscoefall(Xw, [tree₁ tree₂ tree₁])
    @test_throws AssertionError getbasiscoefall(Xw, BitVector([0,1,0]))
    @test_throws AssertionError getbasiscoefall(Xw, BitVector([1,1,1,1]))
    @test_throws AssertionError getbasiscoefall(Xw, maketree(8,2,:dwt))
    @test_throws AssertionError getbasiscoefall(zeros(4,4,2), maketree(4,2,:dwt))
    @test_throws ArgumentError getbasiscoefall(zeros(4,2,2), maketree(4,2,:dwt))
    @test_throws AssertionError getbasiscoefall(Xw, [tree₁ tree₂ tree₁])
    @test_throws AssertionError getbasiscoefall(Xw, BitMatrix([0 0;1 0;0 1]))
    @test_throws AssertionError getbasiscoefall(Xw, BitMatrix([1 1;1 1;1 1;1 1]))
    @test_throws AssertionError getbasiscoefall(Xw, [maketree(8,2,:dwt) maketree(8,2,:dwt)])
    @test_throws AssertionError getbasiscoefall(zeros(4,4,2), BitMatrix([1 1;1 1;1 1]))
    @test_throws ArgumentError getbasiscoefall(zeros(4,2,2), BitMatrix([1 1;1 1;1 1]))

    @test nodelength(8,2) == 2

    @test Utils.main2depthshift(10,4) == [0,0,2,2,10]
    @test Utils.main2depthshift(5,5) == [0,1,1,5,5,5]
    @test_throws AssertionError Utils.main2depthshift(8,3)
    @test_throws AssertionError Utils.main2depthshift(8,2)

    # TODO: @test packet()

    tree = maketree(4,1,:dwt)
    @test coarsestscalingrange(zeros(4), tree) == 1:2
    @test coarsestscalingrange(4, tree) == 1:2
    @test coarsestscalingrange(zeros(4,3), tree, true) == (1:4, 2)
    @test coarsestscalingrange(4, tree, true) == (1:4, 2)
    @test_throws AssertionError coarsestscalingrange(5, tree, true)
    @test_throws AssertionError coarsestscalingrange(5, tree, false)

    @test finestdetailrange(zeros(4), tree) == 3:4
    @test finestdetailrange(4, tree) == 3:4
    @test finestdetailrange(zeros(4,3), tree, true) == (1:4, 3)
    @test finestdetailrange(4, tree, true) == (1:4, 3)
    @test_throws AssertionError finestdetailrange(5, tree, true)
    @test_throws AssertionError finestdetailrange(5, tree, false)

    @test getrowrange(8,2) == 1:4
    @test getrowrange(8,3) == 1:4
    @test getrowrange(8,4) == 5:8
    @test getrowrange(8,5) == 5:8
    @test_throws AssertionError getrowrange(8,86)
    
    @test getcolrange(8,2) == 1:4
    @test getcolrange(8,3) == 5:8
    @test getcolrange(8,4) == 1:4
    @test getcolrange(8,5) == 5:8
    @test_throws AssertionError getcolrange(8,86)
end

@testset "Traverse Trees" begin
    @test isvalidtree(zeros(4,4), maketree(4,4,2,:full))
    @test isvalidtree(zeros(4,4), maketree(4,4,2,:dwt))
    @test !isvalidtree(zeros(4,4), BitVector([0,1,0,0,0]))
    @test !isvalidtree(zeros(4,4), trues(4))

    @test getchildindex(3,:left) == 6
    @test getchildindex(3,:right) == 7
    @test getchildindex(3,:topleft) == 10
    @test getchildindex(3,:topright) == 11
    @test getchildindex(3,:bottomleft) == 12
    @test getchildindex(3,:bottomright) == 13
    @test_throws AssertionError getchildindex(3,:fail)

    @test getparentindex(4,:binary) == 2
    @test getparentindex(5,:binary) == 2
    @test getparentindex(10,:quad) == 3
    @test getparentindex(11,:quad) == 3
    @test getparentindex(12,:quad) == 3
    @test getparentindex(13,:quad) == 3
    @test_throws AssertionError getparentindex(15, :fail)

    qleaf = falses(21); qleaf[[3,4,5,6,7,8,9]] .= 1
    @test getleaf(maketree(4,2,:dwt),:binary) == BitVector([0,0,1,1,1,0,0])
    @test getleaf(maketree(4,4,2,:dwt),:quad) == qleaf
    @test_throws AssertionError getleaf(maketree(4,2,:dwt),:fail)
    @test_throws AssertionError getleaf(maketree(4,4,2,:dwt),:binary)
    @test_throws AssertionError getleaf(BitVector([0,1,0]),:binary)
    @test_throws AssertionError getleaf(BitVector([0,1]),:binary)
    @test_throws AssertionError getleaf(maketree(4,2,:dwt),:quad)
    @test_throws AssertionError getleaf(BitVector([0,1,1,1,1]),:quad)
    @test_throws AssertionError getleaf(BitVector([0,1]),:quad)

    @test maketree(zeros(4,4)) == trues(5)
    @test maketree(zeros(4,4),:dwt) == BitVector([1,1,0,0,0])
    @test maketree(4,4,2) == trues(5)
    @test maketree(4,4,2,:dwt) == BitVector([1,1,0,0,0])
    @test_throws AssertionError maketree(4,4,3,:dwt)
    @test_throws AssertionError maketree(4,4,2,:fail)

    @test getdepth(5,:binary) == 2
    @test getdepth(5,:quad) == 1
    @test_throws AssertionError getdepth(5,:fail)
    @test_throws AssertionError getdepth(0,:binary)

    @test gettreelength(8) == 7
    @test gettreelength(8,8) == 21
    @test gettreelength(8,16) == 21
end

@testset "Metrics" begin
    x₀ = ones(5)
    x = 2*ones(5)
    @test relativenorm(x, x₀) == 1
    @test relativenorm(x, x₀, 1.0) == 1
    @test psnr(x, x₀) == 0
    @test snr(x, x₀) == 0
    @test ssim(x, x₀) == assess_ssim(x, x₀)
end

@testset "Dataset" begin
    x = [1, 0, 0, 0]
    @test duplicatesignals(x, 2, 1) == [1 0; 0 1; 0 0; 0 0]
    @test duplicatesignals(x, 2, 1, true) != duplicatesignals(x, 2, 1)
    @test duplicatesignals(x, 2, 1, true, 0.5) != duplicatesignals(x, 2, 1)

    @test length(generatesignals(:blocks, 5)) == 32
    @test length(generatesignals(:bumps, 5)) == 32
    @test length(generatesignals(:doppler, 5)) == 32
    @test length(generatesignals(:heavisine, 5)) == 32
    @test length(generatesignals(:quadchirp, 5)) == 32
    @test length(generatesignals(:mishmash, 5)) == 32
    @test_throws ArgumentError generatesignals(:fail, 5)

    @test typeof(ClassData(:tri, 5, 5, 5)) == ClassData
    @test typeof(ClassData(:cbf, 5, 5, 5)) == ClassData
    @test_throws ArgumentError ClassData(:fail, 5, 5, 5)

    @test size(generateclassdata(ClassData(:tri, 5, 5, 6))[1]) == (32,16)
    @test size(generateclassdata(ClassData(:cbf, 5, 5, 5))[1]) == (128,15)
    @test_nowarn generateclassdata(ClassData(:tri, 5, 5, 5), true)
end