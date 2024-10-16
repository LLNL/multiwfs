using multiwfs
using DSP: freq

vk = VonKarman()
psd_from_timeseries = psd(von_karman_turbulence(10_000, offset=1.0), 1000)
begin
    fr = freq(psd_from_timeseries)[2:end]
    plot_psd(fr, power(psd_from_timeseries)[2:end], normalize=false)
    plot_psd!(fr, psd_von_karman.(freq(psd_from_timeseries)[2:end], Ref(vk)), normalize=false)
end