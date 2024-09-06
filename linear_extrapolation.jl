######--- Linear extrapolation map ---######

# Extrapolation of single configuration

function lin_ext(log_taus::Vector{Float64}, log_probs::Vector{Float64}, steps::Int64, epsilon::Float64)

    if log_probs[1] != -Inf
        criterion_list = [(log_probs[i+1] - log_probs[i])/(log_taus[i+1] - log_taus[i]) for i in 1:(length(log_taus) - 1)]
        lin_range = findall(x -> abs(x + 2) <= epsilon, criterion_list)
        indices = vcat(lin_range, last(lin_range) + 1)
        
        log_taus_lin = log_taus[indices]
        log_probs_lin = log_probs[indices]

        M = ones(length(log_taus_lin),2)
        M[:,2] = log_taus_lin
        v = log_probs_lin
        coefs = M\v
        
        stepsize = log_taus[2] - log_taus[1]
        log_taus_ext = [last(log_taus) + stepsize * i for i in 1:steps]
        log_probs_ext = [coefs[1] + coefs[2]*(last(log_taus) + stepsize * i) for i in 1:steps]

        log_taus_new = vcat(log_taus, log_taus_ext)
        log_probs_new = vcat(log_probs, log_probs_ext)

        return [log_taus_new, log_probs_new]

    else
        stepsize = log_taus[2] - log_taus[1]
        log_taus_ext = [last(log_taus) + stepsize * i for i in 1:steps]
        log_probs_ext = [-Inf for i in 1:steps]

        log_taus_new = vcat(log_taus, log_taus_ext)
        log_probs_new = vcat(log_probs, log_probs_ext)

        return [log_taus_new, log_probs_new]
   
    end
end

# Extrapolation of vectorized return probability #

function etp(rp_vec::Array{Float64, 2}, tau_range::Vector{Float64}, steps::Int64, epsilon::Float64)

    N = size(rp_vec)[1]
    log_taus = log.(10, tau_range)
    T_ext = zeros(N, length(log_taus) + steps)

    for i in 1:N
        log_probs = log.(10, rp_vec[i,:])
        log_etp = lin_ext(log_taus, log_probs, steps, epsilon)[2]
        etp = exp10.(log_etp)
        T_ext[i,:] = etp
    end

    return T_ext        
    
end