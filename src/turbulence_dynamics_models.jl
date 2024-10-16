using Optim
using LinearAlgebra: eigvals

"""
Returns the PSD of an AR process X_t = ∑_i ϕi X_(t-i).
"""
function psd_ar(f, ϕ)
	denom = 1 - sum(ϕk * exp(-im * 2π * f * k) for (k, ϕk) in enumerate(ϕ))
	return 1 / abs2(denom)
end

function A_ar(ϕ)
	N = length(ϕ)
	A = zeros(N, N)
	A[1,:] = ϕ
	for i in 2:N
		A[i,i-1] = 1
	end
	A
end

function turbulence_ar2_loss(ϕ, turbulence, fr)
	if !(-1 <= ϕ[2] <= 1 - abs(ϕ[1]))
		return Inf
	end
	psd_ar_v = psd_ar.(fr, Ref(ϕ))
	log_diff = log10.(psd_ar_v ./ maximum(psd_ar_v)) - log10.(turbulence ./ maximum(turbulence))
	return sum(abs2, log_diff ./ fr)
end

function best_ar_coeffs(turbulence, fr)
    res = Optim.optimize(ϕ -> turbulence_ar_loss(ϕ, turbulence, fr), [0.1, -0.1], NelderMead())
    return Optim.minimizer(res)
end

export A_ar, best_ar_coeffs