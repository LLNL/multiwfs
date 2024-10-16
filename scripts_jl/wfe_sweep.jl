using multiwfs
using multiwfs: Hrej
using Plots
using ProgressMeter
using Base.Threads
using Dierckx
using NPZ

sys = AOSystem(1000.0, 1.0, 0.01, 0.999, 10, "high", 1.0)
search_gain!(sys)
gains_high = npzread("data/gainmap_loopfreq_1000.0_ftype_high.npy")
gains_low = npzread("data/gainmap_loopfreq_1000.0_ftype_low.npy")

function cumulative_wfe(sys, fr=0.1:0.1:500.0)
    dwfe = (f -> abs2(Hrej(sys, f)) * f^(-11/3)).(fr)
    return cumsum(dwfe .* step(fr))
end

function wfe_sweep(sys, gains; f_cutoffs = 0.1:0.1:100.0, delays = 0.0:0.1:1.0, fr=0.1:0.1:500.0)
    wfes = zeros(length(f_cutoffs), length(delays), length(fr));
    @showprogress @threads for (i, fc) in collect(enumerate(f_cutoffs))
        for (j, d) in enumerate(delays)
            tsys = AOSystem(sys.f_loop, d, gains[i,j], sys.leak, sys.fpf, sys.filter_type, fc)
            wfes[i,j,:] = cumulative_wfe(tsys, fr)
        end
    end
    f_cutoffs, delays, fr, wfes
end

function plot_wfe_sweep(f_cutoffs, delays, wfes; t="", kwargs...)
    p = plot(xlabel="Cutoff frequency (Hz)", ylabel="Relative WFE", title=t; kwargs...)
    c = palette([:blue, :red], length(delays))
    for (i, (r, d)) in enumerate(zip(eachcol(wfes), delays))
        plot!(f_cutoffs, r, label="$d frames", c=c[i])
    end
    p
end

sys.filter_type = "high"
f_cutoffs, delays, fr, wfes_high = wfe_sweep(sys, gains_high);
sys.filter_type = "low"
f_cutoffs, delays, fr, wfes_low = wfe_sweep(sys, gains_low);

begin
    ftype = "high" # high or low
    wfes = eval(Meta.parse("wfes_$ftype"))
    delay_idx = 1 # set the delay for the whole plot
    skip_every = 50
    p = plot(xlabel="Frequency (Hz)", ylabel="Relative cumulative WFE", legend=:outertopright, title="$ftype pass filter, delay = $(delays[delay_idx]) frames, f_cutoff (Hz) on legend", titlefontsize=12, yscale=:log10, xscale=:log10, ylim=(minimum(wfes), maximum(wfes)))
    n_f_cutoff = length(f_cutoffs) ÷ skip_every
    c = palette([:blue, :red], n_f_cutoff)
    for i in 1:n_f_cutoff
        plot!(fr, wfes[i*skip_every,delay_idx,:], c=c[i], label="$(f_cutoffs[i*skip_every])")
    end
    p
end