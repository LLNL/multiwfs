using Base: prod
using StaticArrays

# much of this is already in DSP.jl
# but I'm re-implementing it for better control/understanding

function f2s(f)
    return 1im * 2.0 * π * f
end

struct ZPKFilter{Nz,Np,Nm}
    z::SVector{Nz,ComplexF64}
    p::SVector{Np,ComplexF64}
    k::Float64
    prev_x::MVector{Nm,ComplexF64}
    prev_y::MVector{Nm,ComplexF64}

    function ZPKFilter(z::AbstractArray, p::AbstractArray, k::Number)
        Nz, Np = length(z), length(p)
        Nm = max(Nz, Np)
        x = @MVector zeros(ComplexF64,Nm)
        y = @MVector zeros(ComplexF64,Nm)
        new{Nz,Np,Nm}(z, p, k, x, y)
    end

    function ZPKFilter(z::Number, p::Number, k::Number)
        ZPKFilter([z], [p], k)
    end
end

function output!(zpkf::ZPKFilter{Nz,Np,Nm}, x_n) where {Nz,Np,Nm}
    y_n = 0.0 + 0.0im # intermediate results can be complex but final ones shouldn't be!
    
    for i in 1:Nm
        k = (i == Nm ? zpkf.k : 1)
        if i < min(Nz, Np)
            y_n = zpkf.p[i] * zpkf.prev_y[i] + k * x_n - k * zpkf.z[i] * zpkf.prev_x[i]
        elseif i < Nz
            # zero, no pole
            y_n = k * x_n - k * zpkf.z[i] * zpkf.prev_x[i]
        else
            # pole, no zero
            y_n = zpkf.p[i] * zpkf.prev_y[i] + k * zpkf.prev_x[i]
        end
        zpkf.prev_x[i] = x_n
        zpkf.prev_y[i] = y_n
        x_n = y_n
    end
    return real(y_n)
end

function reset!(zpkf::ZPKFilter{Nz,Np,Nm}) where {Nz,Np,Nm}
    zpkf.prev_x[:] = zeros(Nm)
    zpkf.prev_y[:] = zeros(Nm)
end

function transfer_function(zpkf::ZPKFilter, s)
    z = exp(s)
    return zpkf.k * prod((z - zv) for zv in zpkf.z) / prod((z - p) for p in zpkf.p)
end

function ar1_coeff(f_cutoff, f_loop)
    return exp(-2π * f_cutoff / f_loop)
end

function ar1_filter(f_cutoff, f_loop, filter_type)
    α = ar1_coeff(f_cutoff, f_loop)
    if filter_type == "high"
        return ZPKFilter(1, α, α)
    elseif filter_type == "low"
        return ZPKFilter(0, α, 1 - α)
    end
end

export ar1_filter, ZPKFilter, transfer_function, output!, reset!, f2s
