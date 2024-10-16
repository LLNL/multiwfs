using multiwfs
using Plots
using Roots
using ProgressMeter
using Base.Threads
using ColorSchemes

sys = AOSystem(1000.0, 1.0, 0.01, 0.999, 10, "high", 1.0)
search_gain!(sys)
gains = gain_map(sys)

function zero_db_bandwidth_sweep(sys, gains; f_cutoffs = 0.1:0.1:100.0, delays = 0.0:0.1:1.0)
    zero_db_bandwidths = zeros(length(f_cutoffs), length(delays));
    @showprogress @threads for (i, fc) in collect(enumerate(f_cutoffs))
        for (j, d) in enumerate(delays)
            tsys = AOSystem(sys.f_loop, d, gains[i,j], sys.leak, sys.fpf, sys.filter_type, fc)
            zero_db_bandwidths[i,j] = zero_db_bandwidth(tsys)
        end
    end
    f_cutoffs, delays, zero_db_bandwidths
end

function plot_zero_db_bandwidth_sweep(sys, gains)
    f_cutoffs, delays, zero_db_bandwidths = zero_db_bandwidth_sweep(sys, gains)
    plot_zero_db_bandwidth_sweep(f_cutoffs, delays, zero_db_bandwidths)
end

function plot_zero_db_bandwidth_sweep(f_cutoffs, delays, zero_db_bandwidths; t="")
    p = plot(xlabel="Cutoff frequency (Hz)", ylabel="Zero dB bandwidth (Hz)", title=t)
    c = palette([:blue, :red], length(delays))
    for (i, (r, d)) in enumerate(zip(eachcol(zero_db_bandwidths), delays))
        plot!(f_cutoffs, r, label="$d frames", c=c[i])
    end
    p
end


sys.filter_type = "high"
f_cutoffs, delays, zero_db_bandwidths_high = zero_db_bandwidth_sweep(sys, gains)
sys.filter_type = "low"
f_cutoffs, delays, zero_db_bandwidths_low = zero_db_bandwidth_sweep(sys, gains)
plot_zero_db_bandwidth_sweep(f_cutoffs, delays,min.(zero_db_bandwidths_low, zero_db_bandwidths_high))

plot(
    plot_zero_db_bandwidth_sweep(f_cutoffs, delays, zero_db_bandwidths_low, t="LPF"),
    plot_zero_db_bandwidth_sweep(f_cutoffs, delays, zero_db_bandwidths_high, t="HPF")
)

