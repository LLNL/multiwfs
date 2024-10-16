using ControlSystems: are, Discrete
using multiwfs
using LinearAlgebra: I
using Plots
using Distributions

begin
    f_loop = 1000.0
    Av1 = real.(A_vib(20.0/f_loop, 0.01))
    Av2 = real.(A_vib(0.1/f_loop, 0.1))
    A_ol = block_diag(Av1, Av2)
end

begin
    pls = []
    for highfreq_cost in [1e4, 1e2, 1, 0.01]
        pl = plot(legend=:topleft)
        ws = exp10.(-8:2:-2)
        for (i, w) in enumerate(ws)
            n_input_history = 3
            L = zeros(n_input_history, n_input_history)
            for i in 1:(n_input_history-1)
                L[i+1,i] = 1
            end
            A = block_diag(L, Av1, Av2)
            B = [-1 0 0]
            C = [-1 0 0 1 0 1 0]
            B̃ = [1 0 0 0 0 0 0]'
            Ccost = [1 0 0 highfreq_cost 0 1 0]
            fr = exp10.(-4:0.01:log10(f_loop/2))
            s = 2π * im * fr ./ f_loop
            W, V = zeros(size(A)), hcat(8.0...)
            W[n_input_history+1:end,n_input_history+1:end] = w * I(4)
            Q = Ccost' * Ccost
            R = zeros(1,1)
            K = kalman_gain(A, C, W, V)
            G = lqr_gain(A, B̃, Q, R)
            lqgf = lqg_tf(s, A, B̃, C, K, G)
            plot!(fr, abs2.(lqgf), xscale=:log10, yscale=:log10, label="w = $w", xlabel="Frequency (Hz)", ylabel="|ETF|²", title="HF cost = $highfreq_cost", legend=(highfreq_cost == 1e4 ? :bottomleft : nothing))
        end
        push!(pls, pl)
    end
    plot(pls...)
end