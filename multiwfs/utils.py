import numpy as np
from scipy.signal import windows, welch

def f2s(f):
	return 1j * 2.0 * np.pi * f

def genpsd(tseries, dt, nseg=4):
	nperseg = 2**int(np.log2(tseries.shape[0]/nseg)) #firstly ensures that nperseg is a power of 2, secondly ensures that there are at least nseg segments per total time series length for noise averaging
	window = windows.hann(nperseg)
	freq, psd = welch(tseries, fs=1./dt,window=window, noverlap=nperseg*0.25,nperseg=nperseg, detrend=False,scaling='density')
	freq, psd = freq[1:], psd[1:] #remove DC component (freq=0 Hz)
	return freq, psd

def get_freq(f_loop, N):
	min_freq = f_loop / (2 * N)
	return np.arange(min_freq, f_loop / 2 + min_freq, min_freq)

def rms(data, axis=None):
	"""
	Computes the root-mean-square of `data`.
	"""
	if axis is None:
		axis = tuple(i for i in range(len(data.shape)))
	return np.std(data - data.mean(axis=axis, keepdims=True), axis=axis)

# https://stackoverflow.com/questions/13214809/pretty-print-2d-list
def pretty_print(matrix):
	s = [[str(e) for e in row] for row in matrix]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print('\n'.join(table))