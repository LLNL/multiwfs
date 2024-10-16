using Plots
using DSP

function nyquist_plot!(sys; mark_gm_pm=true, label="Nyquist plot", kwargs...)
    success = :green
    nyquist_contour, gm, gm_point, pm, pm_point = nyquist_and_margins(sys)
    plot!(real(nyquist_contour), imag(nyquist_contour), xlim=(-1.1,1.1), ylim=(-1.1,1.1), aspect_ratio=:equal, legend=:outertopright, label=label; kwargs...)
    phasegrid = range(-π, π, length=500)
    xunit, yunit = cos.(phasegrid), sin.(phasegrid)
    if !is_stable(gm, pm)
        success = :red
    end
    if mark_gm_pm
        vline!([-1/2.5], ls=:dash, label="Gain margin cutoff", color=:grey)
        scatter!([real(gm_point)], [imag(gm_point)], label="Gain margin = $(round(gm, digits=2))", color=:grey)
        plot!([-2,0,-2], [-2,0,2], ls=:dash, label="Phase margin cutoff", color=4)
        if !isnothing(pm_point)
            scatter!([real(pm_point)], [imag(pm_point)], label="Phase margin = $(round(pm, digits=2))", color=4)
        end
    end
    plot!(xunit, yunit, ls=:dash, label=nothing, color=success)
end

function nyquist_plot(sys; kwargs...)
    p = plot()
    nyquist_plot!(sys; kwargs...)
    p
end

function plot_psd(f, p; normalize=true, kwargs...)
    pl = plot()
    plot_psd!(f, p; normalize=normalize, kwargs...)
    pl
end

function plot_psd!(f, p; normalize=true, kwargs...)
    if normalize
        p /= p[1]
    end
    plot!(f, p, xscale=:log10, yscale=:log10, xlabel="Frequency (Hz)", ylabel="Power"; kwargs...)
end

function plot_psd(psd::DSP.Periodograms.Periodogram; kwargs...)
    f, p = freq(psd)[2:end], power(psd)[2:end]
    plot_psd(f, p; kwargs...)
end

function plot_psd!(psd::DSP.Periodograms.Periodogram; kwargs...)
    f, p = freq(psd)[2:end], power(psd)[2:end]
    plot_psd!(f, p; kwargs...)
end

function plot_psd_p!(f, p; kwargs...)
    plot!(f, p, xscale=:log10, yscale=:log10, xlabel="Frequency (Hz)"; kwargs...)
end

export nyquist_plot, nyquist_plot!, plot_psd_p!, plot_psd, plot_psd!