using multiwfs
using Plots
using Distributions
using DSP
import multiwfs: Hrej

function Hrej(systems::Vector{AOSystem}, f)
    ol = sum(Hol(sys, f) for sys in systems)
    return 1 / (1 + ol)
end

f_loop = 1000.0
f_cutoff = 50.0
ar1_low = ar1_filter(f_cutoff, f_loop, "low")
ar1_high = ar1_filter(f_cutoff, f_loop, "high")
# parameters are:
# f_loop, frame_delay, gain, leak, fpf
sys_low = AOSystem(f_loop, 0.1, 1.3, 0.999, 1, ar1_low)
sys_high = AOSystem(f_loop, 1.0, 0.61, 0.999, 1, ar1_high)

p2 = begin
    nyquist_plot(sys_low, mark_gm_pm=false, label="LPF", color=:purple)
    nyquist_plot!(sys_high, mark_gm_pm=false, label="HPF", color=:royalblue4)
    nyquist_plot!([sys_low, sys_high], label="LPF+HPF", color=:cadetblue)
end

f = 0.032:0.032:500.0

p1 = begin
    plot(legend=:bottomright, xticks=[1e-1, 1e0, 1e1, 1e2], yticks=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], ylabel="|Rejection transfer function|²")
    plot_psd_p!(f, abs2.(Hrej.(Ref(sys_low), f)), label="LPF", color=:purple)
    plot_psd_p!(f, abs2.(Hrej.(Ref(sys_high), f)), label="HPF", color=:royalblue4)
    plot_psd_p!(f, abs2.(Hrej.(Ref([sys_high, sys_low]), f)), label="LPF+HPF", color=:cadetblue)
end
