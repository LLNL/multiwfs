using multiwfs
using multiwfs: rms, Hfilter, Hcont
using Plots
using Distributions: MvNormal, Normal
using LinearAlgebra: diag, diagm, I
using Base.GC: gc
using DSP: freq, power
using StatsBase: mean

begin
    f_loop = 1000.0
    freq_low = 0.0079
    damp_low = 1.0
    freq_high = 2.7028
    damp_high = 0.2533
    log_lf_cost = 1.7217
    log_lf_process_noise = -8.0
    log_hf_cost = -0.6807
    log_hf_process_noise = -8.0
    Av1 = A_vib(freq_high/f_loop, damp_high)
    Av2 = A_vib(freq_low/f_loop, damp_low)
    A_ar1 = [0.995 0; 1 0]
    L = A_DM(2)
    Ã = block_diag(L, A_ar1, Av1, Av2)
    C̃ = [0 -1 0 1 0 1 0 1]
    D̃ = [1 0 0 0 0 0 0 0]' 
    B = [0; 0; 1; 0; exp10(log_hf_process_noise); 0; exp10(log_lf_process_noise); 0]
    Pw = hcat(1...)
    W = B * Pw * B'
    V = hcat(1...)
    K̃ = kalman_gain(Ã, C̃, W, V)
    Vv = [0 -1 0 1 0 exp10(log_hf_cost) 0 exp10(log_lf_cost)]
    Q = Vv' * Vv
    R = zeros(1,1)
    L = lqr_gain(Ã, D̃, Q, R)
    lqg = LQG(Ã, D̃, C̃, K̃, L)
end

begin
    N = size(Ã, 1)
    x, x̂, x_ol = zeros(N), zeros(N), zeros(N)
    x += B * rand(Normal())
    y = C̃ * x
    ikca = (I - K̃ * C̃) * Ã
    ikcd = (I - K̃ * C̃) * D̃
    nsteps = 1_000_000
    y, y_ol = zeros(nsteps), zeros(nsteps)
    for i in 1:nsteps
        x_ol = Ã * x_ol
        x = Ã * x + D̃ * L * x̂
        w = B * rand(Normal())
        x += w
        x_ol += w
        x̂ = ikca * x̂ + K̃ * C̃ * x + ikcd * L * x̂ # Kalman update
        y[i] = (C̃ * x)[1]
        y_ol[i] = (C̃ * x_ol)[1]
    end
    gc()
    plot(y_ol, label="OL, RMSE = $(round(rms(y_ol .- mean(y_ol)), digits=3))")
    plot!(y, label="CL, RMSE = $(round(rms(y .- mean(y)), digits=3))")
end

plot_psd(psd(y_ol, 1000), normalize=false)
plot_psd!(psd(y, 1000), normalize=false)

fr, y_psd = genpsd(y, 1000)
fr, yol_psd = genpsd(y_ol, 1000)
etf2 = y_psd ./ yol_psd
etf_analytic = 1 ./ (1 .+ transfer_function.(Ref(lqg), 2π .* im .* fr ./ f_loop))
plot(fr, etf2, xscale=:log10, yscale=:log10, label="Time-domain ETF", xlabel="Frequency (Hz)", ylabel="|ETF|²", legend=:bottomright)
plot!(fr, abs2.(etf_analytic), label="Analytic ETF")

f_cutoff = 0.005
ar1_low = ar1_filter(f_cutoff, f_loop / 10, "low")
sys_low = AOSystem(f_loop, 1.0, 0.1, 0.9999999, 10, ar1_low)
search_gain!(sys_low)

fr = exp10.(-2:0.001:log10(f_loop/2))

begin
    zr = exp.(2π * im * fr ./ f_loop)
    Cfast = z -> transfer_function(lqg, log(z))
    Cslow = z -> (real(log(z) / (2π * im)) < 1/20) ? (Hfilter(sys_low, log(z)) * Hcont(sys_low, log(z))) : 1.0
    R = 10
    p1 = plot(
        fr,
        phi_to_X.(zr, Cfast, Cslow, R) .|> abs2,
        xscale=:log10, yscale=:log10, 
        xlabel="Frequency (Hz)", ylabel="|ETF|²",
        label="|X/Φ|²",
        xticks=exp10.(-4:2), yticks=exp10.(-10:2:2),
        legend=:bottomright, lw=2
    )
    plot!(fr, Lfast_to_X.(zr, Cfast, Cslow, R) .|> abs2, label="|X/Lfast|²", color=2, lw=2)
    plot!(fr, Lslow_to_X.(zr, Cfast, Cslow, R) .|> abs2, label="|X/Lslow|²", color=3, lw=2)
    p2 =  plot(
        fr,
        phi_to_X.(zr, Cfast, Cslow, R) .|> abs2,
        xscale=:log10, yscale=:log10, 
        xlabel="Frequency (Hz)", ylabel="|ETF|²",
        label="|X/Φ|²",
        xticks=exp10.(-4:2), yticks=exp10.(-10:2:2),
        legend=:bottomright, lw=2
    )
    plot!(fr, Nfast_to_X.(zr, Cfast, Cslow, R) .|> abs2, label="|X/Nfast|²", ls=:dash, lw=2, color=2)
    plot!(fr, Nslow_to_X.(zr, Cfast, Cslow, R) .|> abs2, label="|X/Nslow|²", ls=:dash, lw=2, color=3)
    plot(p1, p2, size=(500, 300))
    Plots.savefig("figures/lqgfirst_fiveetfs.pdf")
end