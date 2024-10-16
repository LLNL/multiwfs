function plant(z, Cfast, Cslow, R)
    zinv = 1/z
    fast_term = Cfast(z) * zinv
    slow_term = Cslow(z) * 1/R * sum(zinv^k for k in 1:R)
    return zinv * (fast_term + slow_term)
end

function phi_to_X(z, Cfast, Cslow, R)
    return 1 / (1 + plant(z, Cfast, Cslow, R))
end

function Nfast_to_X(z, Cfast, Cslow, R)
    return -(1/z) * Cfast(z) / (1 + plant(z, Cfast, Cslow, R))
end

function Nslow_to_X(z, Cfast, Cslow, R)
    return -(1/z) * Cslow(z) / (1 + plant(z, Cfast, Cslow, R))
end

function Lfast_to_X(z, Cfast, Cslow, R)
    return -(1/z^2) * Cfast(z) / (1 + plant(z, Cfast, Cslow, R))
end

function Lslow_to_X(z, Cfast, Cslow, R)
    return (1 + (1/z^2) * Cfast(z)) / (1 + plant(z, Cfast, Cslow, R))
end

export plant, phi_to_X, Nfast_to_X, Nslow_to_X, Lfast_to_X, Lslow_to_X