mutable struct AOSystem{F}
    f_loop::Float64
    Ts::Float64
    frame_delay::Float64
    τ::Float64
    gain::Float64
    leak::Float64
    fpf::Int64
    control_filter::F 

    function AOSystem(f_loop, frame_delay, gain, leak, fpf, control_filter)
        Ts = 1 / f_loop
        frame_delay = floor(frame_delay) + round((frame_delay-floor(frame_delay))*fpf)/fpf
        τ = frame_delay * Ts
        new{typeof(control_filter)}(f_loop, Ts, frame_delay, τ, gain, leak, fpf, control_filter)
    end
end

function update_frame_delay!(sys, frame_delay)
    sys.frame_delay = floor(frame_delay) + round((frame_delay-floor(frame_delay))*sys.fpf)/sys.fpf
    sys.τ = sys.frame_delay * sys.Ts
end

function Hwfs(ao::AOSystem, s)
    return (1.0 - exp(-ao.Ts * s)) / (ao.Ts * s)
end

function Hzoh(ao::AOSystem, s)
    return Hwfs(ao, s)
end

function Hlag(ao::AOSystem, s)
    return exp(-ao.τ * s)
end

function Hint(ao::AOSystem, s)
    return ao.gain / (1.0 - exp(-ao.Ts * s)) # integrator
end

function Hlint(ao::AOSystem, s)
    return ao.gain / (1.0 - ao.leak * exp(-ao.Ts * s)) # leaky integrator
end

function Hcont(ao::AOSystem, s)
    return Hlint(ao, s)
end

function Hcont(ao::AOSystem{LQG}, s)
    tf = transfer_function(ao.control_filter, s ./ ao.f_loop)
    return tf / (1 + tf)
end

function Hfilter(ao::AOSystem, s)
    transfer_function(ao.control_filter, s ./ ao.f_loop)
end

function Hfilter(ao::AOSystem{LQG}, s)
    1
end

function Hol(ao::AOSystem, s::Complex)
    return Hwfs(ao, s) * Hlag(ao, s) * Hcont(ao, s) * Hfilter(ao, s) * Hzoh(ao, s)
end

function Hol_unfiltered(ao::AOSystem, f::Real)
    s = f2s(f)
    return Hwfs(ao, s) * Hlag(ao, s) * Hcont(ao, s) * Hzoh(ao, s)
end

function Hol(ao::AOSystem, f)
    s = f2s(f)
    return Hwfs(ao, s) * Hlag(ao, s) * Hcont(ao, s) * Hfilter(ao, s) * Hzoh(ao, s)
end

function Hrej(ao::AOSystem, f)
    return 1.0 / (1.0 + Hol(ao, f))
end

function Hcl(ao::AOSystem, f)
    Hol_val = Hol(ao, f)
    return Hol_val / (1.0 + Hol_val)
end

function Hn(ao::AOSystem, f)
    return Hcl(ao, f) / Hwfs(ao, f2s(f))
end

export AOSystem, Hol, Hol_unfiltered, Hcl, Hrej