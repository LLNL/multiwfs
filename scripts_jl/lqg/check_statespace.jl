using DSP: freq, power
using LinearAlgebra: I
using Plots
using multiwfs
using Distributions
using SciPy
using Base.GC: gc

begin
    σ = 1e0
    noise_dist = Normal(0, σ)
    Nstep = 1000_000
    f_loop = 1000.0
    times = 0.0:(1/f_loop):(Nstep/f_loop)
    Av1 = real.(A_vib(10.0/f_loop, 0.2))
    A = Av1
    B = reshape([1 0], (2,1))
    C = reshape([1 0], (1,2))
    fr = freq(psd(rand(Nstep), f_loop))[2:end]
    s = 2π * im * fr ./ f_loop
    dstf = 1 ./ abs2.(1 .- A[1,1] * exp.(-s) - A[1,2] * exp.(-2s))
    x = zeros(2)
    ys = zeros(Nstep)
    for i in 1:Nstep
        global x = A*x
        global x[1] += rand(noise_dist)
        ys[i] = (C*x)[1]
    end
    olp = psd(ys, f_loop)
    factor = 8 * f_loop * length(fr) / Nstep
    p = plot_psd(fr, factor * power(olp)[2:end], normalize=false, label="OL PSD, time-domain", legend=:bottomleft, color=1)
    plot!(fr, dstf, xscale=:log10, yscale=:log10, label="Analytic OL PSD", color=:black, ls=:dash)
    gc()
    p
end

σ = 1e0
f_loop = 1000
noise_dist = Normal(0, σ)
white_noise = rand(noise_dist, 10_000)
psd_white_noise = psd(white_noise, f_loop)
plot(freq(psd_white_noise)[2:end], power(psd_white_noise)[2:end], xscale=:log10, yscale=:log10)
1 / sum(power(psd_white_noise))

# divide by (sum(windows^2) / periodogram length)