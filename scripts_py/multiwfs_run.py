# %%
import multiwfs
# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiwfs import AOSystem, get_freq

f_loop = 1000.0 # Hz
sys = AOSystem(f_loop=f_loop, frame_delay=0.5, gain=0.2, leak=0.99, filter_type="high", filter_cutoff=30)
freq = get_freq(f_loop, 4096)
# %%
sys.plot_frequency_response(freq)
sys.plot_nyquist(freq)
# %%
s = 2 * np.pi * 1j * freq
sys.gain = 0.2
sys.leak = 0.999
sys.filter_cutoff = 50
Hol_val = sys.Hol(freq)
phase_margin_point = np.abs(np.real(Hol_val)**2+np.imag(Hol_val)**2-1)
phase_margin_ind = np.argmin(phase_margin_point)
pm = float(round(np.angle(Hol_val[phase_margin_ind]) % np.pi * (180 / np.pi), 2))
plt.loglog(freq, np.abs(sys.Hrej(freq)))
plt.xlabel("Frequency (Hz)")
plt.title(f"{sys.filter_type=}, {pm=} deg")
# %%
