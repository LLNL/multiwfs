using ControlSystems: are, Discrete
using DSP: freq, power
using multiwfs
using multiwfs: block_diag
using LinearAlgebra: I
using Plots
using Distributions: Normal, MvNormal

begin
    σ = 1e-2
    noise_dist = Normal(0, σ)
    Nstep = 50_000
    f_loop = 1000.0
    times = 0.0:(1/f_loop):(Nstep/f_loop)
    Av1 = real.(A_vib(50.0/f_loop, 0.05))
    # A = block_diag(Av1, zeros(1,1))
    A = Av1
    B = reshape([1 0], (2,1))
    C = reshape([1 0], (1,2))
    fr = freq(psd(rand(Nstep), f_loop))[2:end]
    s = 2π * im * fr ./ f_loop
    W, V = 1e-10 * I(2), zeros(1,1)
    Q = C' * C
    R = 1e-8 * I(1)
    Pobs = real.(are(Discrete(1/f_loop), A', C', W, V))
    K = Pobs * C' * inv(C * Pobs * C' + V)
    Pcon = real.(are(Discrete(1/f_loop), A, B, Q, R))
    L = -inv(R + B' * Pcon * B) * (B' * Pcon * A)
    dstf = 1e-2 * (σ/2)^2 ./ abs2.(1 .- A[1,1] * exp.(-s) - A[1,2] * exp.(-2s)) 
    lqgf = lqg_controller_tf.(s, Ref(A), Ref(B), Ref(C), Ref(K), Ref(L))
    x = [1; 0]
    xcon = copy(x)
    ys, ycons = Float64[], Float64[]
    for _ in 1:Nstep
        noise = rand(noise_dist)
        x = A*x
        x[1] += noise
        xcon = (A+B*L)*xcon
        xcon[1] += noise
        push!(ys, (C*x)[1])
        push!(ycons, (C*xcon)[1])
    end
    olp = psd(ys, f_loop)
    clp = psd(ycons, f_loop)
    p = plot_psd(fr, power(olp)[2:end], normalize=false, label="OL PSD, time-domain", legend=:left, color=1)
    plot!(fr, dstf, xscale=:log10, yscale=:log10, label="Analytic OL PSD", color=1, ls=:dash)
    plot_psd!(fr, power(clp)[2:end], normalize=false, label="CL PSD, time-domain", color=2)
    plot!(fr, abs2.(lqgf), xscale=:log10, yscale=:log10, label="Analytic CL PSD", color=2, ls=:dash)
    etf_td = (power(clp) ./ power(olp))[2:end]
    plot_psd!(fr, etf_td, normalize=false, label="RTF, time-domain", color=3)
    plot!(fr, abs2.(lqgf ./ dstf), label="Analytic RTF", color=3, ls=:dash)
    p
end

# Next steps:
# 1. Implement the Looze control scheme
# 2. Add in the Kalman filter part so it's just an end-to-end controller TF
# 3. Put the resulting analytic ETF on sliders for the four parameters