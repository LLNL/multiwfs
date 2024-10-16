### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6e63bc5c-6721-11ef-0e75-89cf19a9f817
begin
	using Pkg
	Pkg.activate("..")
	using multiwfs
	using Plots
	using LinearAlgebra: I
	using Base.GC: gc
	using PlutoUI
	using PlutoUI: combine
	using QuadGK
	using DataInterpolations
	using Roots
	using Polynomials
end

# ╔═╡ e4f22ab9-7e18-437a-a3d2-d60f26d7c4ff
using Symbolics

# ╔═╡ c57e0875-5d37-4693-9e6e-f02125cf4c0c
function design(params::Vector, lows::Vector, highs::Vector)
	
	return combine() do Child
		inputs = [
			md""" $(name): $(
				Child(name, Slider(l:1e-4:h))
			)"""
			
			for (name, l, h) in zip(params, lows, highs)
		]
		
		md"""
		#### OL design
		$(inputs)
		"""
	end
end;

# ╔═╡ 98416ad3-5460-4b52-bf3b-073707a2e053
begin
	f_loop = 1000.0
	fr = exp10.(-4:0.01:log10(f_loop/2))
	cutoff_idx = findfirst(x -> x > f_loop/20, fr)
	fr_slow = fr[1:cutoff_idx]
end;

# ╔═╡ 4622ed0c-6e04-4d18-a636-6316538c8b6c
vk = VonKarman();

# ╔═╡ 2cb452db-4559-49ee-9dcf-1ca164c433c8
begin
	freq_low = 0.0079
	damp_low = 1.0
	freq_high = 2.7028
	damp_high = 0.2533
	log_lf_cost = 1.7217
	log_lf_process_noise = -8.0
	log_hf_cost = -0.6807
	log_hf_process_noise = -8.0
end;

# ╔═╡ fc73834e-37f1-4e3a-98e8-1d0e990fb3f2
"""@bind pars design(
	["log_lf_cost", "log_lf_process_noise", "log_hf_cost", "log_hf_process_noise"],
	[-8.0, -8.0, -8.0, -8.0],
	[8.0, 8.0, 8.0, 8.0]
)""";

# ╔═╡ 9484c00f-2dde-4ddb-8b30-55947784025c
begin
	f_cutoff = 1
	ar1_low = ar1_filter(f_cutoff, f_loop / 10, "low")
	sys_low = AOSystem(f_loop, 1.0, 0.1, 0.999999, 10, ar1_low)
	search_gain!(sys_low)
	lpf_ol = Hol.(Ref(sys_low), fr_slow)
	low_etf = Hrej.(Ref(sys_low), fr_slow)
	sys_low.gain 
end

# ╔═╡ 0ae2ddcf-1341-412c-8560-7af718eb6e8b
function lqg_design_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)
	Av1 = A_vib(freq_high/f_loop, damp_high)
    Av2 = A_vib(freq_low/f_loop, damp_low)
	A_ar1 = [0.995 0; 1 0]
    L = A_DM(2)
    Ã = block_diag(L, A_ar1, Av1, Av2)
    C̃ = [0 -1 0 1 0 1 0 1]
    D̃ = [1 0 0 0 0 0 0 0]' 
    B = [0; 0; 1; 0; exp10(log_hf_process_noise); 0; exp10(log_lf_process_noise); 0]
    Pw = hcat(1...)
    W = B * Pw * B'
    V = hcat(1...)
    K̃ = kalman_gain(Ã, C̃, W, V)
    Vv = [0 -1 0 1 0 exp10(log_hf_cost) 0 exp10(log_lf_cost)]
    Q = Vv' * Vv
    R = zeros(1,1)
    L = lqr_gain(Ã, D̃, Q, R)
	return Ã, D̃, C̃, K̃, L
end;

# ╔═╡ b84a3b69-c406-422c-a881-8b7858b6c5f3
function lqg_etf_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)
	Ã, D̃, C̃, K̃, L = lqg_design_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)
    s = 2π * im .* fr / f_loop
    z = exp.(s)
    zinvs = 1 ./ z
    gc()
	return 1 ./ (1 .+ lqg_controller_tf(Ã, D̃, C̃, K̃, L, zinvs))
end;

# ╔═╡ 551dccd9-7348-4c18-8f9e-d2e65fba8a6f
begin
	lqg_etf = lqg_etf_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)
	lqg_etf_norm = abs2.(lqg_etf)
	lqg_etf_norm_interp = CubicSpline(lqg_etf_norm, fr, extrapolate=true)
	residual_error = sqrt(quadgk(f -> psd_von_karman(f, vk) * lqg_etf_norm_interp(f), 0, 500)[1])
	hpf_ol = 1 ./ lqg_etf .- 1
	plot(
		fr,
		lqg_etf_norm,
		xscale=:log10, yscale=:log10, xlabel="Frequency (Hz)", ylabel="|ETF|²", xticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
		title="HPF residual error = $(round(residual_error, digits=3)) rad", ylims=(1e-8, 1e2), label="High-speed ETF",
		legend=:bottomleft
	)
	plot!(
		fr_slow,
		abs2.(low_etf),
		xscale=:log10, yscale=:log10, label="Low-speed ETF",
	)
	combined_etf_norm = abs2.(1 ./ (1 .+ hpf_ol[1:cutoff_idx] .+ lpf_ol))
	append!(combined_etf_norm, ones(length(fr) - length(fr_slow)))
	plot!(
		fr,
		combined_etf_norm,
		lw=2, color=4, label="Combined ETF"
	)
	plot!([1], [1], color=3, label="Input PSD (right y-axis)")
	plot!([1], [1], color=5, label="CL PSD (right y-axis)")
	yaxis2 = twinx()
	plot!(yaxis2, fr, psd_von_karman.(fr, Ref(vk)), xscale=:log10, yscale=:log10, color=3, legend=nothing, ylabel="Power (rad²/Hz)", yticks=[1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], ylims=(1e-8, 1e2))
	plot!(yaxis2, fr, psd_von_karman.(fr, Ref(vk)) .* combined_etf_norm, color=5)
	hline!([1], ls=:dash, color=:black, label=nothing)
	# vline!([freq_low], label="freq_low = $(freq_low) Hz, damp_low = $(damp_low)", ls=:dash, color=:black)
	# vline!([freq_high], label="freq_high = $(freq_high) Hz, damp_high = $(damp_high)", ls=:dash, color=:black)
end

# ╔═╡ 2b18cef8-5b13-48f9-9bda-db17b9c9b24b
lqg_design_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)[4]

# ╔═╡ 4cd07eca-976c-4230-8031-45a2004bbbbd
begin
	A_low_AR = [0.995 0; 1 0]
	L = A_DM(2)
	A_low = block_diag(L, A_low_AR)
	D_low = [1 0 0 0]'
	C_low = [0 -1 0 1]
	B_low = [0; 0; 1; 0]
	W_low = B_low * hcat(1...) * B_low'
	V_low = hcat(1...)
	K_low = kalman_gain(A_low, C_low, W_low, V_low)
	Vv_low = [0 -1 0 1]
	Q_low = Vv_low' * Vv_low
	R_low = zeros(1,1)
	L_low = lqr_gain(A_low, D_low, Q_low, R_low)
	s = 2π * im .* fr_slow / (f_loop / 10)
    z = exp.(s)
    zinvs = 1 ./ z
	L
	low_etf_lqg = 1 ./ (1 .+ lqg_controller_tf(A_low, D_low, C_low, K_low, L_low, zinvs))
end;

# ╔═╡ a364d840-a783-4b59-8c27-f3eba952eadb
@variables x;

# ╔═╡ a36790ce-36d4-4ff5-94fa-4929b9745743
begin
	A, D, C, K, L_l = lqg_design_from_params(freq_high, damp_high, freq_low, damp_low, log_lf_cost, log_lf_process_noise, log_hf_cost, log_hf_process_noise, f_loop)
	analytic_tf = simplify(1 / (1 + lqg_controller_tf(A, D, C, K, L_l, 1/x)))
end;

# ╔═╡ 29793343-5b1c-45f6-b37b-8ab1473f4019
function extract_coeffs(symbolic_args)
	i = 0
	coeffs = []
	while true
		new_coeff = Symbolics.coeff(symbolic_args, x^i)
		if (i > 0) & (new_coeff ≈ 0)
			return coeffs
		end
		push!(coeffs, new_coeff)
		i += 1
	end
end

# ╔═╡ 95f5db8f-ea75-4433-8617-7745d802ed64
begin
	numerator_coeffs = Float64.(extract_coeffs(simplify(analytic_tf.val.arguments[1], expand=true)))
	denominator_coeffs = Float64.(extract_coeffs(simplify(analytic_tf.val.arguments[2], expand=true)))
	normalizer = numerator_coeffs[end]
	numerator_coeffs ./= normalizer
	denominator_coeffs ./= normalizer
end

# ╔═╡ 35cf0b26-a642-46f8-af34-e4ffed8f886d
lqg_zeros = roots(Polynomial(numerator_coeffs))

# ╔═╡ 958c1cd8-ae47-43af-b888-0fea03d7ac17
lqg_poles = roots(Polynomial(denominator_coeffs))

# ╔═╡ 260a0349-332d-4696-b839-d4bd4dd096c0
begin
	plot(aspect_ratio=:equal)
	scatter!(real.(lqg_zeros), imag.(lqg_zeros), label="Zeros")
	scatter!(real.(lqg_poles), imag.(lqg_poles), label="Poles")
end

# ╔═╡ f9ceaa2d-104a-487a-a52e-924ae9e51834
ZPKFilter(lqg_zeros, lqg_poles, 1.0)

# ╔═╡ Cell order:
# ╟─6e63bc5c-6721-11ef-0e75-89cf19a9f817
# ╟─c57e0875-5d37-4693-9e6e-f02125cf4c0c
# ╠═98416ad3-5460-4b52-bf3b-073707a2e053
# ╠═4622ed0c-6e04-4d18-a636-6316538c8b6c
# ╠═2cb452db-4559-49ee-9dcf-1ca164c433c8
# ╟─fc73834e-37f1-4e3a-98e8-1d0e990fb3f2
# ╟─551dccd9-7348-4c18-8f9e-d2e65fba8a6f
# ╠═9484c00f-2dde-4ddb-8b30-55947784025c
# ╠═0ae2ddcf-1341-412c-8560-7af718eb6e8b
# ╠═b84a3b69-c406-422c-a881-8b7858b6c5f3
# ╠═2b18cef8-5b13-48f9-9bda-db17b9c9b24b
# ╟─4cd07eca-976c-4230-8031-45a2004bbbbd
# ╠═e4f22ab9-7e18-437a-a3d2-d60f26d7c4ff
# ╠═a364d840-a783-4b59-8c27-f3eba952eadb
# ╠═a36790ce-36d4-4ff5-94fa-4929b9745743
# ╠═29793343-5b1c-45f6-b37b-8ab1473f4019
# ╠═95f5db8f-ea75-4433-8617-7745d802ed64
# ╠═35cf0b26-a642-46f8-af34-e4ffed8f886d
# ╠═958c1cd8-ae47-43af-b888-0fea03d7ac17
# ╠═260a0349-332d-4696-b839-d4bd4dd096c0
# ╠═f9ceaa2d-104a-487a-a52e-924ae9e51834
