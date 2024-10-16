using NPZ
using Roots

function zero_crossings(f, x, y)
    crossing_regions = findall(x -> x < 0, y[1:end-1] .* y[2:end])
    zero_vals = zeros(length(crossing_regions))
    for (i, crossing_idx) in enumerate(crossing_regions)
        zero_vals[i] = fzero(f, (x[crossing_idx], x[crossing_idx+1]))
    end
    zero_vals
end

"""
Takes in a complex number z and returns its angle relative to -1 in either direction, in degrees.
-1-1im and -1+1im should both return 45.
"""
function angle_relative_to_minus1(z)
    return 180 - abs(rad2deg(angle(z)))
end

function Hol(systems::Vector{AOSystem}, f)
    return sum(Hol(sys, f) for sys in systems)
end

function nyquist_and_margins(sys)
    if sys isa Vector
        f_loop = maximum(s.f_loop for s in sys)
    else
        f_loop = sys.f_loop
    end
    f = (0.001, f_loop / 2 + 0.001)
    gm, gm_point, pm, pm_point = Inf, nothing, 180, nothing
    oneside_freq = range(minimum(f), maximum(f), length=2001)
    linfreq = vcat(-reverse(oneside_freq), oneside_freq)
    nyquist_contour = Hol.(Ref(sys), linfreq)
    imag_axis_crossings = zero_crossings(freq -> imag(Hol(sys, freq)), linfreq, imag.(nyquist_contour))
    gm_candidate_points = Hol.(Ref(sys), imag_axis_crossings)
    if length(gm_candidate_points) > 0
        gm_point = minimum(real(x) for x in gm_candidate_points if (!isnan(x)) & (real(x) > -1))
        gm = -1 / real(gm_point)
    end
    unit_circle_crossings = zero_crossings(freq -> abs2(Hol(sys, freq)) - 1, linfreq, abs2.(nyquist_contour) .- 1)
    pm_candidate_points = Hol.(Ref(sys), unit_circle_crossings)
    if length(pm_candidate_points) > 0
        pm_point = pm_candidate_points[argmin(angle_relative_to_minus1.(pm_candidate_points))]
        pm = angle_relative_to_minus1(pm_point)
    end
    return nyquist_contour, gm, gm_point, pm, pm_point
end

function margins(sys)
    _, gm, _, pm, _ = nyquist_and_margins(sys)
    return (gm=gm, pm=pm)
end

function is_stable(sys)
    try
        _, gm, _, pm, _ = nyquist_and_margins(sys)
        return is_stable(gm, pm)
    catch
        return false
    end
end

function is_stable(gm, pm)
    return gm > 2.5 && pm >= 45.0
end

function search_gain!(sys)
    sys.gain = 1.0
    gain_min, gain_max = 1e-15, 1.0
    while gain_max - gain_min > 1e-15
        if is_stable(sys)
            gain_min = sys.gain
        else
            gain_max = sys.gain
        end
        sys.gain = (gain_min + gain_max) / 2
    end
    if !is_stable(sys)
        sys.gain = sys.gain - 1e-15
    end
    sys.gain
end

function zero_db_bandwidth(sys)
    try
        # try to solve via root-finding
        return find_zero(f -> abs(Hol(sys, f)) - 1.0, (0.1, 500.0))
    catch
        # fall back to grid evaluation
        f = 0.1:0.1:500.0
        abs_Hol_val = abs.(Hol.(Ref(sys), f))
        fstart = argmax(abs_Hol_val)
        fend = findlast(abs_Hol_val .<= 1.0)
        return f[fstart:fend][findfirst(abs_Hol_val[fstart:fend] .<= 1.0)]
    end
end    

function ar1_gain_map(sys, filter_type; f_cutoffs = 0.1:0.1:100.0, delays = 0.0:0.1:1.0, save=true)
    gain_map = zeros(length(f_cutoffs), length(delays));
    @showprogress @threads for (i, fc) in collect(enumerate(f_cutoffs))
        for (j, d) in enumerate(delays)
            tsys = AOSystem(sys.f_loop, d, sys.gain, sys.leak, sys.fpf, ar1_filter(fc, sys.f_loop, filter_type))
            search_gain!(tsys)
            gain_map[i,j] = tsys.gain
        end
    end
    if save
        npzwrite("data/gainmap_loopfreq_$(sys.f_loop)_ftype_$filter_type.npy", gain_map)
    end
    gain_map
end

export ar1_gain_map, search_gain!, zero_db_bandwidth, get_nyquist_contour, margins