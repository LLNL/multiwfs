include("filter_setup.jl")

begin
    f_cutoff = 3.0
    # parameters are:
    # f_loop, frame_delay, gain, leak, fpf, filter_type, filter_cutoff
    sys_low = AOSystem(200.0, 1.0, 1.0, 0.999, 1, ar1_low)
    sys_high = AOSystem(200.0, 0.1, 1.0, 0.999, 10, ar1_high)
    #search_gain!(sys_low)
    sys_low.gain = 0.3
    #search_gain!(sys_high)

    f = 0.01:0.001:100

    Hol_vals = Hol_unfiltered.(Ref(sys_low), f) .+ Hol.(Ref(sys_high), f)
    Hrej_vals = @. 1 / (1 + Hol_vals)

    # plot(f, abs2.(Hrej_vals), xscale=:log10, yscale=:log10, xticks=[1e-2, 1e-1, 1e0, 1e1, 1e2], yticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], xlabel="Frequency (Hz)", ylabel="Error transfer function", label=nothing)
    plot(f, abs2.(Hol_unfiltered.(Ref(sys_low), f)), xscale=:log10, yscale=:log10, xlabel="Frequency (Hz)", ylabel="Open-loop TF unfiltered")
end

