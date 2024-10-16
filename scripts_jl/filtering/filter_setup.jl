using multiwfs
using SciPy
using Plots

f_loop = 200.0
f_cutoff = 3.0
ar1_high = ar1_filter(f_cutoff, f_loop, "high")
ar1_low = ar1_filter(f_cutoff, f_loop, "low")
ch_rp = -20*log10(0.95)
ch_omegan = [f_cutoff / (f_loop / 2)] # critical frequencies, nyquist = 1
@assert 0 < ch_omegan[1] < 1
cheb2_high = ZPKFilter(signal.cheby1(2, ch_rp, ch_omegan, "highpass", output="zpk")...)
cheb4_high = ZPKFilter(signal.cheby1(4, ch_rp, ch_omegan, "highpass", output="zpk")...)
cheb2_low = ZPKFilter(signal.cheby1(2, ch_rp, ch_omegan, "lowpass", output="zpk")...)
cheb4_low = ZPKFilter(signal.cheby1(4, ch_rp, ch_omegan, "lowpass", output="zpk")...)