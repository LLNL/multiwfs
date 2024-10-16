using DSP
using StatsBase: mean

function psd(x, f_loop)
    noverlap = 2^(Int(floor(log2(length(x)/4)))-2)
    n = div(length(x), 8)
    return welch_pgram(x, n, noverlap; fs=f_loop)
end

function integrator_control(sys, open_loop, gain, leak, update_every; hpf_gain=0.0, delay_frames=1)
    N = length(open_loop)
    closed_loop = zeros(N)
    closed_loop[1] = open_loop[1]
    command = 0.0
    average_buffer = []
    for i in 2:N
        push!(average_buffer, closed_loop[i-1])
        y_n = output!(sys.control_filter, closed_loop[i-delay_frames])[1]
        if i % update_every == 0
            command = leak * command - gain * mean(average_buffer)
            average_buffer = []
        end
        # hpf is stuck to update_every = 1
        command -= hpf_gain * y_n
        closed_loop[i] = open_loop[i] + command
    end
    closed_loop
end

export psd, integrator_control