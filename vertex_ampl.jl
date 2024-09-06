######--- Conditions on the spins ---######

# SU2-recoupling condition #

function SU2_cond(j1::Float64, j2::Float64)

    if mod(j1 + j2, 1.0) == 0
    
        return 1

    else

        return 0
        
    end
    
end

# Riemannian EPRL-condition #

function EPRL_cond(j::Float64, gamma::Rational{Int64})

    j = rationalize(j)
    j_p = (abs(1 + gamma)//2) * j
    j_m = (abs(1 - gamma)//2) * j

    if float(mod(j_p, 1//2)) == 0.0 && float(mod(j_m, 1//2)) == 0.0
        
        return 1

    else 
        
        return 0

    end
    
end

######--- Dressed vertex amplitude ---######

function Ampl_cont(alpha::Float64,G::Float64,γ::Float64,Lambda::Float64,j1::Float64,j2::Float64,j3::Float64)

    res = VAmp(G,γ,Lambda,j1,j2,j3) * sqrt(EAmp(γ,j1,j1,j1)) * sqrt(EAmp(γ,j2,j2,j2)) * EAmp(γ,j1,j2,j3)^3
    res *= FAmp(alpha,γ,j1)^(3. /2.) * FAmp(alpha,γ,j2)^(3. /2.) * FAmp(alpha,γ,j3)^(3.)

    res = real(res)

    return res
    
end

# Dressed vertex amplitude with Riemannian EPRL-condition # 

function Ampl(alpha::Float64,G::Float64,γ::Float64,Lambda::Float64,j1::Float64,j2::Float64,j3::Float64)

    if EPRL_cond(j1, rationalize(γ)) == 1 && EPRL_cond(j2, rationalize(γ)) == 1 && EPRL_cond(j3, rationalize(γ)) == 1

        res = VAmp(G,γ,Lambda,j1,j2,j3) * sqrt(EAmp(γ,j1,j1,j1)) * sqrt(EAmp(γ,j2,j2,j2)) * EAmp(γ,j1,j2,j3)^3
        res *= FAmp(alpha,γ,j1)^(3. /2.) * FAmp(alpha,γ,j2)^(3. /2.) * FAmp(alpha,γ,j3)^(3.)

        res = real(res)

        return res

    else 
        
        return 0.0

    end
end


function Ampl_resc(alpha::Float64,G::Float64,γ::Float64,Lambda::Float64,j1::Float64,j2::Float64,j3::Float64)

    if EPRL_cond(j1, rationalize(γ)) == 1 && EPRL_cond(j2, rationalize(γ)) == 1 && EPRL_cond(j3, rationalize(γ)) == 1

        res = VAmp(G,γ,Lambda,j1,j2,j3) * sqrt(EAmp(γ,j1,j1,j1)) * sqrt(EAmp(γ,j2,j2,j2)) * EAmp(γ,j1,j2,j3)^3
        res *= FAmp(alpha,γ,j1)^(3. /2.) * FAmp(alpha,γ,j2)^(3. /2.) * FAmp(alpha,γ,j3)^(3.)

        res = real(res)

        res *= (1 - γ^2)^(6.0*alpha - 9.0/2.0)/(pi^3)

        return res

    else 
        
        return 0.0

    end
end


function Ampl_resc_tensor(alpha::Float64,G::Float64,gamma::Float64,Lambda::Float64, jmax::Float64)

    Ns = Int64[]
    Ms = Int64[]
    As = Float64[]

    N = Int(2 * jmax)
    nzeros = [i for i in 1:N if EPRL_cond(0.5*i, rationalize(gamma)) == 1]
    for n in nzeros
        for m in nzeros
            j = 0.5*n
            k = 0.5*m
            A = Ampl_resc(alpha, G, gamma, Lambda, j, j, k)
            push!(Ns, n)
            push!(Ms, m)
            push!(As, A)
        end
    end

    Ampl_tensor = sparse(Ns, Ms, As, N, N)

    return Ampl_tensor

end



######--- 1-periodic ---######

# 1-periodic vectorized amplitude #

function Ampl_vector(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    N = Int(2*(j_max - j_min) + 1)
    v = zeros(N^2)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N for a_0 in A_i, a_1 in A_i.-1])
    
    for s in nzeros        
        i = 0.5*nconvert1(s-1, N)[1] + j_min
        j = 0.5*nconvert1(s-1, N)[2] + j_min
        v[s] = Ampl(alpha, G, γ, Lambda, i, i, j)
    end

    return v
    
end


######--- 2-periodic ---######

# 2-periodic amplitude #

function Am16(alpha::Float64, G::Float64,γ::Float64,Lambda::Float64,
    jin::Float64, jmid::Float64, k1::Float64, k2::Float64)

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2) 

        result = 1

        result *= Ampl(alpha,G,γ,Lambda,jin,jmid,k1)^8

        result *= Ampl(alpha,G,γ,Lambda,jmid,jin,k2)^8

    else 

        result = 0.0
    
    end

    return result

end

function Am16_cont(alpha::Float64, G::Float64,γ::Float64,Lambda::Float64,
    jin::Float64, jmid::Float64, k1::Float64, k2::Float64)

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2) 

        result = 1

        result *= Ampl_cont(alpha,G,γ,Lambda,jin,jmid,k1)^8

        result *= Ampl_cont(alpha,G,γ,Lambda,jmid,jin,k2)^8

    else 

        result = 0.0
    
    end

    return result

end

# 2-periodic vectorized amplitude #

function Am16_vector(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    N = Int(2*(j_max - j_min) + 1)
    v = zeros(N^4)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])

    for s in nzeros
        i = 0.5*nconvert2(s-1, N)[1] + j_min
        j = 0.5*nconvert2(s-1, N)[2] + j_min
        k = 0.5*nconvert2(s-1, N)[3] + j_min
        l = 0.5*nconvert2(s-1, N)[4] + j_min
        v[s] = Am16(alpha, G, γ, Lambda, i, j, k, l)
    end

    return v
    
end


#######--- 3-periodic ---######

# 3-periodic amplitude #

function Am81(alpha::Float64, G::Float64,γ::Float64,Lambda::Float64, jin::Float64, j1::Float64,j2::Float64,
            k1::Float64, k2::Float64,k3::Float64)

    if -1. /sqrt(2) <= (j1 - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (j2 - j1) / (4*k2) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - j2) / (4*k3) <= 1. /sqrt(2) && EPRL_cond(jin, γ) == 1 &&
        EPRL_cond(j1, γ) == 1 && EPRL_cond(j2, γ) == 1 && 
        EPRL_cond(k1, γ) == 1 && EPRL_cond(k2, γ) == 1 &&
        EPRL_cond(k3, γ) == 1 && SU2_cond(jin, j1) == 1 && SU2_cond(j1, j2) == 1 &&
        SU2_cond(j2, jin) == 1

        result = 1

        result *= Ampl(alpha,G,γ,Lambda,jin,j1,k1)^27

        result *= Ampl(alpha,G,γ,Lambda,j1,j2,k2)^27

        result *= Ampl(alpha,G,γ,Lambda,j2,jin,k3)^27

    else 

        result = 0.0

    end

    return result

end

# 3-periodic vectorized amplitude #

function Am81_vector(alpha::Float64, G::Float64, γ::Float64, Lambda::Float64, j_min::Float64, j_max::Float64)

    N = Int(2*(j_max - j_min) + 1)
    v = spzeros(N^6)
    A_i = findall(x -> EPRL_cond(x, γ) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 + a_4*N^4 + a_5*N^5  
    for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1, a_4 in A_i.-1, a_5 in A_i.-1])

    for s in nzeros
        i = 0.5*nconvert3(s-1, N)[1] + j_min
        j = 0.5*nconvert3(s-1, N)[2] + j_min
        k = 0.5*nconvert3(s-1, N)[3] + j_min
        l = 0.5*nconvert3(s-1, N)[4] + j_min
        m = 0.5*nconvert3(s-1, N)[5] + j_min
        n = 0.5*nconvert3(s-1, N)[6] + j_min
        v[s] = Am81(alpha, G, γ, Lambda, i, j, k, l, m, n)
    end

    return v
    
end