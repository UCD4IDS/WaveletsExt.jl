import WaveletsExt.WaveMult: dyadlength, stretchmatrix, ndyad, sft, isft

@testset "Utilities" begin
    # dyadlength
    @test dyadlength(zeros(16)) == dyadlength(16) == 4
    @test_logs (:warn, "Dyadlength n != 2^J") dyadlength(15)

    # stretchmatrix
    i = [1,2,3,4]; j = copy(i)
    @test stretchmatrix(i,j,4,2) == ([1, 4, 7, 8], [1, 4, 7, 8])
    @test_throws AssertionError stretchmatrix(i,j,4,3)
    @test_throws AssertionError stretchmatrix(i,j,4,0)
    @test_throws AssertionError stretchmatrix(i,j,4,-1)
    
    # ndyad
    @test ndyad(1,4,false) == 17:24
    @test ndyad(1,4,true) == 25:32
    @test_throws AssertionError ndyad(5,4,true)
    @test_throws AssertionError ndyad(5,4,false)
    @test_throws AssertionError ndyad(0,4,false)
    @test_throws AssertionError ndyad(0,4,false)
end

@testset "Wavelet Transforms" begin
    # 1D Nonstandard transform
    x = [1,2,-3,4.0]
    wt = wavelet(WT.haar)
    y = [2, 0, 2, -1, 2.1213, 0.7071, 0.7071, 4.9497]
    z = [3.5, 4.5, -1.5, 5.5]
    @test round.(ns_dwt(x, wt), digits = 4) == y
    @test round.(ns_idwt(y, wt), digits = 4) == z
    @test_throws AssertionError ns_dwt(x, wt, 3)
    @test_throws AssertionError ns_dwt(x, wt, 0)
    @test_throws AssertionError ns_idwt(y, wt, 3)
    @test_throws AssertionError ns_idwt(y, wt, 0)

    # Standard form transform
    x = [1 2 -3 4; 2 3 -4 1; 3 4 -1 2; 3 1 -2 3.0]
    y = [4.75 -4.75 0.3536 7.0711;
         1.75 0.25 -1.0607 -1.4142;
         -0.7071 -2.1213 0 -1;
         -1.0607 1.0607 -1.5 1]
    @test round.(sft(x, wt), digits = 4) == y
    @test round.(isft(y, wt), digits = 4) == x
    @test_throws AssertionError sft(x, wt, 3)
    @test_throws AssertionError sft(x, wt, 0)
    @test_throws AssertionError isft(y, wt, 3)
    @test_throws AssertionError isft(y, wt, 0)

    # Build sparse matrices
    x = [1 0 -3 0; 0 3 0 1; 3 0 -1 0; 0 1 0 3.0]
    y = [2 0 0 0 0 0 0 0;
         0 0 0 0 0 0 0 0;
         0 0 0 -2 0 0 0 0;
         0 0 1 1 0 0 0 0;
         0 0 0 0 0 0 1 2;
         0 0 0 0 0 0 -1 2;
         0 0 0 0 1 2 2 -1;
         0 0 0 0 -1 2 2 1.0]
    z = [2 -2 0 2.8284;
         1 1 -1.4142 0;
         2.1213 0.7071 2 -1;
         0.7071 2.1213 2 1]
    y_sparse = sparse(y)
    z_sparse = sparse(z)
    @test round.(mat2sparseform_nonstd(x, wt), digits = 4) == y_sparse
    @test round.(mat2sparseform_std(x, wt), digits = 4) == z_sparse
    @test_throws AssertionError mat2sparseform_nonstd(randn(4,5), wt)
    @test_throws AssertionError mat2sparseform_std(randn(4,5), wt)
end

@testset "Wavelet Multiplication" begin
    M = zeros(4,4)
    for i in axes(M,1)
        for j in axes(M,2)
            M[i,j] = i==j ? 0 : 1/abs(i-j)
        end
    end
    x = [0.5, 1, 1.5, 2]
    wt = wavelet(WT.haar)
    
    # Test on Nonstandard Wavelet Multiplication
    y = [2.4167, 3, 3.25, 2.1667]
    @test round.(nonstd_wavemult(M, x, wt), digits = 4) == y

    # Test on Standard Wavelet Multiplication
    y = [2.4167, 3, 3.25, 2.1667]
    @test round.(std_wavemult(M, x, wt), digits = 4) == y
end