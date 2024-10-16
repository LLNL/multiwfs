using QuadGK, Roots

L0 = 10.0
cutoff_low, cutoff_high = 1e-8, 500.0

total_error = quadgk(f -> (f + (1/L0))^(-11/6), 1e-8, 500.0)[1]
function objective(fc)
    return quadgk(f -> (f + (1/L0))^(-11/6), cutoff_low, fc)[1] - 7/8 * total_error
end

find_zero(objective, (cutoff_low, 500.0), Roots.Brent())
