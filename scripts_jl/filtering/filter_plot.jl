include("filter_setup.jl")

begin
    fr = (6e-4:1e-4:0.5) .* f_loop
    sr = f2s.(fr ./ f_loop)
    plot(fr, abs2.(transfer_function.(Ref(ar1_high), sr)), xscale=:log10, yscale=:log10, ylim=(1e-5, 1.1), xlabel="Frequency (Hz)", ylabel="Power (normalized)", label="AR(1) HPF", xticks=[1e-1, 1e0, 1e1, 1e2])
    plot!(fr, abs2.(transfer_function.(Ref(cheb2_high), sr)), xscale=:log10, yscale=:log10, label="Chebyshev order 2 HPF")
    plot!(fr, abs2.(transfer_function.(Ref(cheb4_high), sr)), xscale=:log10, yscale=:log10, label="Chebyshev order 4 HPF")
    vline!([f_cutoff], label="Cutoff frequency", color=:black, ls=:dash)
end