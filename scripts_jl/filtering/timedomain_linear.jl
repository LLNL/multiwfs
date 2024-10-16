using multiwfs
using Plots
using Distributions
using DSP

include("filter_setup.jl")

# assess Ben's 1 rad rms criterion for AR(1), Cheb2, Cheb4

begin
    f_cutoff = 3.0
    f_loop = 200.0
    ar1_high = ar1_filter(f_cutoff, f_loop, "high")
    no_filter = ZPKFilter(0, 0, 1)
    # parameters are:
    # f_loop, frame_delay, gain, leak, fpf
    sys_high = AOSystem(f_loop, 1.0, 0.45, 0.999, 10, ar1_high)
    sys_test = AOSystem(f_loop, 0.0, 0.3, 0.999, 10, no_filter)
    sys_slow = AOSystem(f_loop / 10, 0.0, 1.0, 0.999, 1, no_filter)
end;

# order 2 should behave similarly to the AR
# combined TF should have a saddle in both 2 and 4
# order 4 should have no improvement
# in the zpk form, each individual filter term should remember its last state
# in the ba form, you need a length n history

# let's make some open-loop turbulence
begin
    N = 50000
    f_loop = 200
    open_loop_t = zeros(N)
    for i in 2:N
        open_loop_t[i] = 0.995 * open_loop_t[i-1] + rand(Normal(0, 0.001))
    end
    const open_loop = open_loop_t
end
multiwfs.reset!(sys_high.control_filter)
plot(integrator_control(sys_high, open_loop, 1.0, 0.999, 10, hpf_gain=0.15, delay_frames=1))

begin
    multiwfs.reset!(sys_high.control_filter)
    ol_psd_p = psd(open_loop, f_loop)
    f, ol_psd = freq(ol_psd_p)[2:end], power(ol_psd_p)[2:end]
    etf_regular = power(psd(integrator_control(sys_test, open_loop, 0.3, 0.999, 1, delay_frames=1), f_loop))[2:end] ./ ol_psd
    etf_slow = power(psd(integrator_control(sys_slow, open_loop, 1.0, 0.999, 10, delay_frames=1), f_loop))[2:end] ./ ol_psd
    etf_filt = power(psd(integrator_control(sys_high, open_loop, 1.0, 0.999, 10, hpf_gain=0.15, delay_frames=1), f_loop))[2:end] ./ ol_psd
end

begin
    plot(title="Estimated ETFs - Slow WFS @ 20 Hz", legend=:bottomright, xticks=[1e-1, 1e0, 1e1, 1e2])
    plot_psd_p!(f, etf_regular, label="Regular g = 0.30", color=:green)
    plot_psd_p!(f, etf_slow, label="Slow gₛ = 1.00", color=2)
    plot_psd_p!(f, etf_filt, label="Slow + Fast-HPF gₛ = 1.00, g = 0.15", color=1)
    plot_psd_p!(f, abs2.(1 ./ (1 .+ Hol_unfiltered.(Ref(sys_test), f))), color=:black,  label=nothing)
    plot_psd_p!(f, abs2.(1 ./ (1 .+ Hol_unfiltered.(Ref(sys_slow), f))), color=:black,  label=nothing)
    plot_psd_p!(f, abs2.(1 ./ (1 .+ Hol_unfiltered.(Ref(sys_slow), f) .+ Hol.(Ref(sys_high), f))), color=:black,  label=nothing)
end

ar1_low = ar1_filter(3.0, f_loop, "low")

begin
    plot(f, abs2.(1 ./ (1 .+ Hol_unfiltered.(Ref(sys_slow), f) .* transfer_function.(Ref(ar1_low), 2π * im * f / f_loop) .+ Hol.(Ref(sys_high), f))), xscale=:log10, yscale=:log10, label="Low + high on the separate WFSs", xticks=[1e-1, 1e0, 1e1, 1e2])
    plot_psd_p!(f, abs2.(1 ./ (1 .+ Hol_unfiltered.(Ref(sys_slow), f) .+ Hol.(Ref(sys_high), f))), label="High pass on the fast WFS", legend=:bottomright)
end
