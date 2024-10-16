# %%
import numpy as np

tf = lambda s: 6 / (s ** 3 + 4 * s ** 2 + 6 * s + 4)
freq = np.arange(0.01, 10000, 0.01)
res = tf(2 * np.pi * 1j * freq)

gain_margin_point = np.abs(np.imag(res) / np.real(res))
gain_margin_ind = np.argmin(gain_margin_point)
print(res[gain_margin_ind])
print(float(-20 * np.log10(-np.real(res[gain_margin_ind]))))
# %%
