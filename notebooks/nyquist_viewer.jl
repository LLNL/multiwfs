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

# ╔═╡ b6f41cbe-42eb-11ef-29db-f7b08d9f8ed0
begin
	using Pkg
	Pkg.activate("..")
	using multiwfs
	using multiwfs: Hrej, Hcl, Hol
	using Plots
	using PlutoUI
	using Revise
end

# ╔═╡ c87e29c0-af46-49c0-8264-fb8c90c7eb76
@bind gain Slider(0.01:0.01:1.2)

# ╔═╡ f5b29214-724b-4e7c-ae72-61c7bda8af72
begin
	f_loop = 200.0
	f_cutoff = 3.0
	ar1_high = ar1_filter(f_cutoff, f_loop, "high")
	
	sys_high = AOSystem(
	    f_loop, 1.0, gain, 0.999, 10,
	    ar1_high
	)
	
	# search_gain!(sys_high)
	nyquist_plot(sys_high)
end

# ╔═╡ Cell order:
# ╠═b6f41cbe-42eb-11ef-29db-f7b08d9f8ed0
# ╟─c87e29c0-af46-49c0-8264-fb8c90c7eb76
# ╠═f5b29214-724b-4e7c-ae72-61c7bda8af72
