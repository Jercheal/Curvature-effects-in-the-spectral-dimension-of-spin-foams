#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# This julia file contains the functions to compute the expectation value of the return probability for a given data set ret_prob. Used in https://arxiv.org/abs/2304.13058 #
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------

### 1-periodic partition function 

function Z1(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    res = sum(Ampl_vector(alpha, G, γ, Lambda, j_min, j_max))

    return res
end

#----------------------------------------------

### 1-periodic partition function with integration 

function Z1_int(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, jmin::Float64, jmax::Float64)

    res = cuhre( (x,f) -> f[1] = (jmax - jmin)^2 * 
    real(Ampl_alt(alpha, G, γ, Lambda, x[1] * (jmax - jmin) + jmin, x[1] * (jmax - jmin) + jmin, x[2] * (jmax - jmin) + jmin)),
    2,1,minevals=1e3,maxevals=1e4
)

return res[1][1]
    
end

#----------------------------------------------

### Expectation value of 1-periodic return probability 

function Expval_P1(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, return_prob1, j_min::Float64, j_max::Float64)

    len_tau_range = length(return_prob1[1,:])
    rp1 = Array(return_prob1)

    res = zeros(Float64, len_tau_range)
    mul!(res, transpose(rp1), Ampl_vector(alpha, G, γ, Lambda, j_min, j_max)) 

    res *= 1/(Z1(alpha, G, γ, Lambda, j_min, j_max))

    return res
    
end

function Expval_P1_fast(dVAmp, return_prob1)

    len_tau_range = length(return_prob1[1,:])
    rp1 = Array(return_prob1)

    res = zeros(Float64, len_tau_range)
    mul!(res, transpose(rp1), dVAmp) 

    res *= 1/(sum(dVAmp))

    return res
    
end

#----------------------------------------------

### Expectation value of 1-periodic return probability with integration 

function Expval_P1_int(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, jmin::Float64, jmax::Float64, tau::Float64)

    integral = cuhre( (x,f) -> f[1] = (jmax - jmin)^2 *
    real(Ampl_alt(alpha, G, γ, Lambda, x[1] * (jmax - jmin) + jmin, x[1] * (jmax - jmin) + jmin, x[2] * (jmax - jmin) + jmin) * 
    return_probability_1_per(x[1] * (jmax - jmin) + jmin, x[2] * (jmax - jmin) + jmin, tau)),
            2,1,minevals=1e4,maxevals=1e5
        )
    res  = integral[1][1]

    res *= 1/(Z1_int(alpha, G, γ, Lambda, jmin, jmax))

    return res
    
end

#----------------------------------------------

### 2-periodic partition function 

function Z2(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    res = sum(Am16_vector(alpha, G, γ, Lambda, j_min, j_max))

    return res
    
end

#----------------------------------------------

### Expectation value of 2-periodic return probability 

function Expval_P2(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, return_prob2, j_min::Float64, j_max::Float64)

    len_tau_range = length(return_prob2[1,:])
    ampl = Am16_vector(alpha, G, γ, Lambda, j_min, j_max)

    res = zeros(Float64, len_tau_range)
    mul!(res, transpose(return_prob2), ampl) 
    
    res *= 1/(sum(ampl))
    
end


#----------------------------------------------

### 3-periodic partition function #

function Z3(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    res = sum(Am81_vector(alpha, G, γ, Lambda, j_min, j_max))

    return res
    
end

#----------------------------------------------

### Expectation value of 3-periodic return probability 

function Expval_P3(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, return_prob3, j_min::Float64, j_max::Float64)

    len_tau_range = length(return_prob3[1,:])

    res = zeros(Float64, len_tau_range)
    mul!(res, transpose(return_prob3), Am81_vector(alpha, G, γ, Lambda, j_min, j_max))  
    
    res *= 1/(Z3(alpha, G, γ, Lambda, j_min, j_max))
    
end