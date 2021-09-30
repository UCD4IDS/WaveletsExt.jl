tree = BitVector([1,1,1])
@test WaveletsExt.Visualizations.treenodes_matrix(tree) == BitArray([1 1; 1 1])
# plot_tfbdry() test
@test typeof(plot_tfbdry(tree)) == Plots.Plot{Plots.GRBackend}
@test_nowarn plot_tfbdry(tree, start=1)
@test_throws AssertionError plot_tfbdry(tree, start=2)
@test_nowarn plot_tfbdry(tree, nodecolor=:red)
x = randn(16,5)
# wiggle() test
@test typeof(wiggle(x)) == Plots.Plot{Plots.GRBackend}
p = plot()
@test typeof(wiggle(p, x)) == Plots.Plot{Plots.GRBackend}
# wiggle!() test
@test typeof(wiggle!(x)) == Plots.Plot{Plots.GRBackend}
@test typeof(wiggle!(p, x)) == Plots.Plot{Plots.GRBackend}
@test_throws ErrorException("Inconsistent taxis dimension!") wiggle!(p, x, taxis=1:5)
@test_throws ErrorException("Inconsistent zaxis dimension!") wiggle!(p, x, zaxis=1:4)
@test_nowarn wiggle!(p, x, sc=0.3)
@test_nowarn wiggle!(p, x, EdgeColor=:red)
@test_nowarn wiggle!(p, x, FaceColor=:red)
@test_nowarn wiggle!(p, x, Overlap=false)
@test_nowarn wiggle!(p, x, Orient=:down)
@test_throws AssertionError wiggle!(p, x, Orient=:fail)
@test_nowarn wiggle!(p, x, ZDir=:reverse)
@test_throws AssertionError wiggle!(p, x, ZDir=:fail)