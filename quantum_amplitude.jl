#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# This julia file contains the functions to compute the dressed quantum vertex amplitude from given data and to compute corresponding expectation values, all being used in https://arxiv.org/abs/2304.13058 #
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

using LinearAlgebra, SparseArrays, CSV, DataFrames

#include("/home/jercheal/Documents/Physics/Codes/Spectral Dimension/basic_frusta.jl")
#include("/home/jercheal/Documents/Physics/Codes/Spectral Dimension/laplacians_v3.jl")
#include("/home/jercheal/Documents/Physics/Codes/Spectral Dimension/vertex_ampl.jl")

function dressed_QVAmp(SU2_QEAmp::Matrix{Float64}, SU2_QVAmp::Matrix{Float64}, alpha::Float64, gamma::Float64)

    n_configs = size(SU2_QEAmp)[1]
    gamma = rationalize(gamma)
    n_max = floor(Int, (n_configs * 2)/(1 +  gamma))
    nzeros = [n for n in 1:n_max if EPRL_cond(n/2.0, gamma) == 1]
    full_QVAmp = zeros(n_max, n_max)

    QEamp = SU2_QEAmp
    QVamp = SU2_QVAmp

    for n in nzeros
        for m in nzeros
            n_plus  = Int((1 + gamma)/2 * n) 
            n_minus = Int((1 - gamma)/2 * n) 
            m_plus  = Int((1 + gamma)/2 * m)
            m_minus = Int((1 - gamma)/2 * m)

            full_QVAmp[n,m] = ((float(n_plus) + 1) * (float(n_minus) + 1) *
                             (float(m_plus) + 1) * (float(m_minus) + 1))^(3.0 * alpha)
            full_QVAmp[n,m] *= QEamp[n_plus, n_plus] * QEamp[n_minus, n_minus]
            full_QVAmp[n,m] *= QEamp[n_plus, m_plus]^(3.0) * QEamp[n_minus, m_minus]^(3.0)
            full_QVAmp[n,m] *= QVamp[n_plus, m_plus] * QVamp[n_minus, m_minus]
        end
    end

    #println("The maximum number of configurations with ", n_configs, " SU2-configs and with gamma = ", gamma, " is ", n_max)

    return full_QVAmp
    
end

function QZ1(dressed_VAmp)

    return sum(dressed_VAmp)
    
end

function QExpval_P1(dressed_VAmp, return_prob1)

    tau_length = length(return_prob1[1,:])
    rp1 = Array(return_prob1)
    res = zeros(Float64, tau_length)

    mul!(res, transpose(rp1), vec(dressed_VAmp))

    res /= QZ1(dressed_VAmp)

    return res

end