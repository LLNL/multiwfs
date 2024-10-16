using multiwfs: psd_ar, psd, plot_psd, plot_psd!
using DSP: freq, power
using ProgressMeter
using FFTW
using StatsBase: std, mean
using Plots

freq1 = 1.0
freq2 = 5.0
f_loop = 1000

nsubframes = 100_000
white_noise = rand(nsubframes) * 2π # phase
grid_v = 0:nsubframes-1
xy = sqrt.((grid_v .- nsubframes/2).^2)
amplitude = 0.01 * ones(length(xy))
feature_indices_low = findall(-freq2 .<= -xy ./ (nsubframes / f_loop) .<= -freq1)
feature_indices_middle = findall(-freq1 .<= xy ./ (nsubframes / f_loop) .<= freq1)
feature_indices_high = findall(freq1 .<= xy ./ (nsubframes / f_loop) .<= freq2)

peak_amp = 1
amplitude[feature_indices_low] .= peak_amp
amplitude[feature_indices_middle] .= 0.01
amplitude[feature_indices_high] .= peak_amp

amplitude[Int(nsubframes/2)] = 0.0 # remove piston
amp = circshift(amplitude, -Int(nsubframes/2))
complex_fourier_domain = amp .* exp.(im .* white_noise)
real_time_domain = real.(ifft(complex_fourier_domain))

noise2signal = 0.01
noise = randn(nsubframes)
noise = noise .* std(real_time_domain) / std(noise) * noise2signal
real_time_domain = real_time_domain .+ noise
real_time_domain *= 8 / std(real_time_domain)
real_time_domain = real_time_domain .- mean(real_time_domain)
plot(real_time_domain[1:1000])

plot_psd(psd(real_time_domain, f_loop))

function past_n_steps(data, n)
    return data[n+1:end], hcat([data[n+1-i:end-i] for i in 1:n]...)'
end

begin
    psd_true = psd(real_time_domain, f_loop)
    fr = freq(psd_true)[2:end]
    p = plot_psd(fr, power(psd(real_time_domain, f_loop))[2:end], label="Source OL PSD", legend=:topright)
    vline!([freq1, freq2], ls=:dash, color=:black, label="Injected frequency peak")
    ks = [1, 2, 3, 4, 5, 10, 20, 30, 500]
    @showprogress for (i, k) in enumerate(ks)
        to_predict, history = past_n_steps(real_time_domain, k)
        ar_coeffs = history' \ to_predict 
        psd_v = psd_ar.(fr ./ f_loop, Ref(ar_coeffs))
        plot_psd!(fr, psd_v, label="Best-fit AR($k) OL PSD", xticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], legend=:topright, c=RGB(1-i/length(ks), i/length(ks),0), yticks=[1e-4, 1e-2, 1e0, 1e2])
    end
    p
end