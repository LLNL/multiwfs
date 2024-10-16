using LinearAlgebra: I
using ControlSystems: are, Discrete
using SciPy: linalg

function A_vib(f_over_f_loop, k)
    ω = 2π * f_over_f_loop
    a₁ = 2 * exp(-k * ω) * cos(ω * sqrt(1 - k^2))
    a₂ = -exp(-2 * k * ω)
    return [a₁ a₂; 1 0]
end

function dynamic_system_tf(s, A, B, C)
    return (C * inv(exp(s)*I - A) * B)[1,1]
end

function kalman_gain(A, C, W, V)
    P = are(Discrete(1), A', C', W, V)
    K = P * C' * inv(C * P * C' + V)
    return K
end

function lqr_gain(A, B, Q, R)
    P = nothing
    try
        P = are(Discrete(1), A, B, Q, R)
    catch 
        P = linalg.solve_discrete_are(A, B, Q, R)
    end
    L = -inv(B' * P * B) * B' * P * A
    return L
end

function A_DM(n_input_history)
    L = zeros(n_input_history, n_input_history)
    for i in 1:(n_input_history-1)
        L[i+1,i] = 1
    end
    L
end

function lqg_controller_tf(A, D, C, K, G, zinvs)
    ikcA = (I - K * C) * A
    ikcD = (I - K * C) * D
    numerator = [(G * inv(I - ikcA * zinv))[1,1] for zinv in zinvs]
	denominator = [(I - G * inv(I - ikcA * zinv) * ikcD * zinv)[1,1] for zinv in zinvs]
    return numerator .* (zinvs .^ 2) ./ denominator
end


export A_vib, A_DM, dynamic_system_tf, lqg_controller_tf, kalman_gain, lqr_gain