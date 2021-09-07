using Wavelets, WaveletsExt
using Plots

n = 2^9;
x₀ = testfunction(n, "Doppler");
x₁ = x₀ + 0.05*randn(n)

# Plot the signal
plot(x₀, lc="black", lw=1, label="Original Signal")
plot!(x₁, lw=0.7, label="Noisy Signal")

# Autocorrelation Wavelet Transform
y = acwt(x₁, wavelet(WT.db4));

# Visualize the decomposition
wiggle(y, sc=0.5)

# Threshold coefficients
threshold!(y, HardTH(), 0.2)

# Perform inverse transfrom
z = iacwt(y)

plot(x₀, lc="black", label="Original")
plot!(x₁, lc="gray", lw=0.7, label="Noisy")
plot!(z, label="Denoised")