using multiwfs
using Symbolics
using Polynomials
using Plots

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

fr = exp10.(-4:0.01:log10(f_loop/2))
sr = 2π .* im * (fr ./ f_loop)
lqg_ol_tf = transfer_function.(Ref(lqg), sr)

sys_high = AOSystem(
    f_loop, 1.0, 1.0, 0.0, 10,
    lqg
)

nyquist_plot(sys_high)

