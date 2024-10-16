using LinearAlgebra: I

include("make_lqg.jl")

struct LQG
    A::Matrix{Float64}
    D::Matrix{Float64}
    C::Matrix{Float64}
    K::Matrix{Float64}
    L::Matrix{Float64}
    ikcA::Matrix{Float64}
    ikcD::Matrix{Float64}

    function LQG(A, D, C, K, L)
        ikcA = (I - K * C) * A
        ikcD = (I - K * C) * D
        new(A, D, C, K, L, ikcA, ikcD)
    end
end

function transfer_function(lqg::LQG, s::Complex)
    zinv = exp(-s)
    numerator = (lqg.L * inv(I - lqg.ikcA * zinv))[1,1]
	denominator = (I - lqg.L * inv(I - lqg.ikcA * zinv) * lqg.ikcD * zinv)[1,1]
    return numerator * (zinv ^ 2) / denominator
end

export LQG, transfer_function