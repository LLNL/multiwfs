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

# ╔═╡ cfb168a2-67f2-11ef-3b76-b9ffa3c20af1
begin
	using Pkg
	Pkg.activate("..")
	using multiwfs
	using multiwfs: psd_ar
	using Plots
	using Polynomials
	using PlutoUI
	using PlutoUI: combine
	using Base.GC: gc
	using DataInterpolations
	using QuadGK
end

# ╔═╡ 58b5e538-6fd4-46c6-bf44-937553c0e331
f_loop = 1000

# ╔═╡ d1bfe27e-5ddd-4136-9758-564762b59b21
fr = exp10.(-4:0.01:log10(f_loop/2))

# ╔═╡ 166ad005-63ef-4d6a-b2e6-4fc8d4bbf81e
function design(params::Vector)
	
	return combine() do Child
		inputs = [
			md""" $(name): $(
				Child(name, Slider(-2:0.001:2))
			)"""
			
			for name in params
		]
		
		md"""
		#### OL design
		$(inputs)
		"""
	end
end;

# ╔═╡ dd6de783-dac7-41c5-b737-53b510d26ff4
@bind p design(
	["a1", "a2", "a3", "a4", "log_cost"]
)

# ╔═╡ b243cc2f-4dcb-4bc7-a40a-d2285ed7a8d5
ϕ = [p.a1, p.a2, p.a3, p.a4]

# ╔═╡ 53a786ef-0f9f-47b7-a846-ef3d09486f3b
p

# ╔═╡ 27197e17-b04d-4bcf-81d0-9cf3cd59913e
stable = all(abs.(roots(Polynomial(vcat([1], -ϕ)))) .> 1)

# ╔═╡ 5163caaf-ad6e-4239-b3f0-5cd1e942e21c
function lqg_design_from_params(ϕ, log_cost, f_loop)
	A_ar = A_DM(4)
	A_ar[1,:] = ϕ
    L = A_DM(2)
    Ã = block_diag(L, A_ar)
    C̃ = [0 -1 0 1 0 0]
    D̃ = [1 0 0 0 0 0]' 
    B = [0; 0; 1; 0; 0; 0]
    Pw = hcat(1...)
    W = B * Pw * B'
    V = hcat(1...)
    K̃ = kalman_gain(Ã, C̃, W, V)
    Vv = [0 -1 0 exp10(log_cost) 0 0]
    Q = Vv' * Vv
    R = zeros(1,1)
    L = lqr_gain(Ã, D̃, Q, R)
	return Ã, D̃, C̃, K̃, L
end

# ╔═╡ 339b6c7b-b090-4042-9357-f68de3dfd923
function lqg_etf_from_params(ϕ, cost, f_loop)
	Ã, D̃, C̃, K̃, L = lqg_design_from_params(ϕ, cost, f_loop)
    s = 2π * im .* fr / f_loop
    z = exp.(s)
    zinvs = 1 ./ z
    gc()
	return 1 ./ (1 .+ lqg_controller_tf(Ã, D̃, C̃, K̃, L, zinvs))
end

# ╔═╡ 300c98b9-d674-44a1-aaaf-bb25072515b3
vk = VonKarman();

# ╔═╡ f9320e91-adef-4f7a-9e68-bf4ed47832a6
begin
	lqg_etf_norm = abs2.(lqg_etf_from_params(ϕ, exp10(p.log_cost), f_loop))
	lqg_etf_norm_interp = CubicSpline(lqg_etf_norm, fr, extrapolate=true)
	residual_error = sqrt(quadgk(f -> psd_von_karman(f, vk) * lqg_etf_norm_interp(f), 0, 500)[1])
	plot(
		fr,
		lqg_etf_norm,
		xscale=:log10, yscale=:log10, xlabel="Frequency (Hz)", ylabel="|ETF|²", xticks=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
		title="residual error = $(round(residual_error, digits=3)) rad", ylims=(1e-4, 1e4), label="ETF",
		legend=:topleft,  c=(stable ? :green : :red)
	)
end

# ╔═╡ Cell order:
# ╠═cfb168a2-67f2-11ef-3b76-b9ffa3c20af1
# ╠═58b5e538-6fd4-46c6-bf44-937553c0e331
# ╠═d1bfe27e-5ddd-4136-9758-564762b59b21
# ╠═b243cc2f-4dcb-4bc7-a40a-d2285ed7a8d5
# ╟─166ad005-63ef-4d6a-b2e6-4fc8d4bbf81e
# ╟─53a786ef-0f9f-47b7-a846-ef3d09486f3b
# ╠═dd6de783-dac7-41c5-b737-53b510d26ff4
# ╠═f9320e91-adef-4f7a-9e68-bf4ed47832a6
# ╟─27197e17-b04d-4bcf-81d0-9cf3cd59913e
# ╠═5163caaf-ad6e-4239-b3f0-5cd1e942e21c
# ╠═339b6c7b-b090-4042-9357-f68de3dfd923
# ╠═300c98b9-d674-44a1-aaaf-bb25072515b3
