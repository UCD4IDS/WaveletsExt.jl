@testset "Indexing" begin
    @test left(1) == 2
    @test right(1) == 3
    @test nodelength(8,2) == 2

    tree = BitVector([1, 1, 1, 1, 0, 1, 0])
    @test getleaf(tree) == BitVector([0,0,0,0,1,0,1,1,1,0,0,1,1,0,0])
end

@testset "Error Rates" begin
    
end