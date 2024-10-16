using SciPy: signal

function block_diag(matrices...)
    matrices = collect(matrices)
    total_dim = 0
    for m in matrices
        @assert ndims(m) == 2
        @assert size(m, 1) == size(m, 2)
        total_dim += size(m, 1)
    end
    types = [typeof(m).parameters[1] for m in matrices]
    if ComplexF64 in types
        t = ComplexF64
    else
        t = Float64
    end
    A = zeros(t, total_dim, total_dim)
    i = 1
    for m in matrices
        k = size(m, 1)
        A[i:(i+k-1), i:(i+k-1)] .= m
        i += k
    end
    A
end

function genpsd(tseries, f_loop, nseg=4)
	nperseg = 2^Int(round(log2(length(tseries)/nseg))) #firstly ensures that nperseg is a power of 2, secondly ensures that there are at least nseg segments per total time series length for noise averaging
	window = signal.windows.hann(nperseg)
	freq, psd = signal.welch(tseries, fs=f_loop,window=window, noverlap=nperseg*0.25,nperseg=nperseg, detrend=false,scaling="density")
	freq, psd = freq[2:end], psd[2:end] #remove DC component (freq=0 Hz)
	return freq, psd
end

export block_diag, genpsd