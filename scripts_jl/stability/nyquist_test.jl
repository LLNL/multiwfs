using multiwfs
include("../filtering/filter_setup.jl")

f_loop = 200.0
f_cutoff = 3.0

sys_high = AOSystem(
    f_loop, 1.0, 0.5, 0.999, 10,
    ar1_high
)

search_gain!(sys_high)
nyquist_plot(sys_high)

