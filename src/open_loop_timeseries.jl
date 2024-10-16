using FFTW
using StatsBase: mean, std
using Plots
using SciPy
using QuadGK

struct VonKarman
    f₀::Float64
    prefactor::Float64

    function VonKarman(D=3, v=10, rms_target=8, f_loop=1000)
        f₀ = v / D
        prefactor = rms_target^2 / quadgk(f -> (f + f₀)^(-11/3), 0, f_loop/2)[1]
        new(f₀, prefactor)
    end
end

function psd_von_karman(f, vk::VonKarman)
    return vk.prefactor * (f + vk.f₀)^(-11/3)
end

function von_karman_turbulence(nframes=1000; fpf=10, f_loop=1000, offset=10/3)
    nsubframes = nframes * fpf
    pl = -11/3 # Kolmogorov?
    white_noise = rand(nsubframes) * 2π # phase
    grid = 0:nsubframes-1
    xy = sqrt.((grid .- nsubframes/2).^2 .+ offset * nsubframes)
    # ask Ben how to change this to von Karman
    amplitude = (xy .+ 1).^(pl/2) # amplitude central value in xy grid is one, so max val is one in power law, everything else lower

    amplitude[Int(nsubframes/2)] = 0.0 # remove piston

    amp = circshift(amplitude, -Int(nsubframes/2)) # is this the same as circshift? Looks like yes
    complex_fourier_domain = amp .* exp.(im .* white_noise)
    real_time_domain = real.(ifft(complex_fourier_domain))

    noise2signal = 0.01
    noise = randn(nsubframes)
    noise = noise .* std(real_time_domain) / std(noise) * noise2signal
    real_time_domain = real_time_domain .+ noise
    real_time_domain *= 8 / std(real_time_domain)
    return real_time_domain .- mean(real_time_domain)
end

export VonKarman, psd_von_karman, von_karman_turbulence