using multiwfs
using Plots

sys = AOSystem(200.0, 1.0, 0.01, 0.999, 10, ar1_filter(30.0, 200.0, "low"))
search_gain!(sys)
nyquist_plot(sys)
zero_db_bandwidth(sys)

