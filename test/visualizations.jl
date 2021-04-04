tree = BitVector([1,1,1])
@test WaveletsExt.Visualizations.treenodes_matrix(tree) == BitArray([1 1; 1 1])
@test typeof(plot_tfbdry(tree)) == Plots.Plot{Plots.GRBackend}
x = randn(16,5)
@test typeof(wiggle(x)) == Plots.Plot{Plots.GRBackend}
@test typeof(wiggle!(x)) == Plots.Plot{Plots.GRBackend}