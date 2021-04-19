@testset "Indexing" begin
    @test left(1) == 2
    @test right(1) == 3
    @test nodelength(8,2) == 2

    tree = BitVector([1, 1, 1, 1, 0, 1, 0])
    @test getleaf(tree) == BitVector([0,0,0,0,1,0,1,1,1,0,0,1,1,0,0]) 

    tree = BitVector([1,0,0])
    x = randn(4)
    @test coarsestscalingrange(x, tree) == 1:2
    @test coarsestscalingrange(4, tree) == 1:2
    @test coarsestscalingrange(x, tree, true) == (1:4, 2)
    @test coarsestscalingrange(4, tree, true) == (1:4, 2)
    @test finestdetailrange(x, tree) == 3:4
    @test finestdetailrange(4, tree) == 3:4
    @test finestdetailrange(x, tree, true) == (1:4, 3)
    @test finestdetailrange(4, tree, true) == (1:4, 3)
end

@testset "Error Rates" begin
    x₀ = ones(5)
    x = 2*ones(5)
    @test relativenorm(x, x₀) == 1
    @test relativenorm(x, x₀, 1.0) == 1
    @test psnr(x, x₀) == 0
    @test snr(x, x₀) == 0
    @test ssim(x, x₀) == assess_ssim(x, x₀)
end

@testset "Generate Signals" begin
    x = [1, 0, 0, 0]
    @test generatesignals(x, 2, 1) == [1 0; 0 1; 0 0; 0 0]
    @test generatesignals(x, 2, 1, true) != generatesignals(x, 2, 1)
    @test generatesignals(x, 2, 1, true, 0.5) != generatesignals(x, 2, 1)
end