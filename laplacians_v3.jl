######--- 1-periodic ---######

# 1-periodic Laplacian and its spectrum #

function spectrum_Laplace_1per_temp(j::Float64, k::Float64, L::Int64)

    V = vol_frusta(j,j,k)
    v1 = vol_3d_cube(j) / dual_len_t(j,j,j,k,k)

    spectrum_array = [-2 * v1/V * (1 - cos(((2*pi)/L) * p)) for p in 0:(L-1)]

    return spectrum_array
    
end

function spectrum_Laplace_1per_temp!(spectrum_array, j::Float64, k::Float64, L::Int64)

    V = vol_frusta(j,j,k)
    v1 = vol_3d_cube(j) / dual_len_t(j,j,j,k,k)

    for p = 0:L-1
        spectrum_array[p+1] = -2 * v1/V * (1 - cos(((2*pi)/L) * p))
    end

    return
    
end

function spectrum_Laplace_1per_spat(j::Float64, k::Float64, L::Int64)
    
    V = vol_frusta(j,j,k)
    w1 = vol_3d_frusta(j,j,k) / dual_len_sp(j,j,k)

    spectrum_array = [-2* w1/V * (1 - cos(((2*pi)/L) * q)) for q in 0:(L-1)]
    
    return spectrum_array
    
end

function spectrum_Laplace_1per_spat!(spectrum_array, j::Float64, k::Float64, L::Int64)
    
    V = vol_frusta(j,j,k)
    w1 = vol_3d_frusta(j,j,k) / dual_len_sp(j,j,k)

    for q = 0:L-1
        spectrum_array[q+1] = -2* w1/V * (1 - cos(((2*pi)/L) * q))
    end
    
    return
    
end

function return_probability_1_per_sum(j::Float64, k::Float64, taus::Vector{Float64}, L::Int64)

    rp1 = zeros(Float64, length(taus))
    spec_t = spectrum_Laplace_1per_temp(j, k, L)
    spec_s = spectrum_Laplace_1per_spat(j, k, L)

    for t in 1:length(taus)
        tau = taus[t]
        v = spec_t .* tau
        expv = exp.(v)
        rp1_temp = sum(expv)
        rp1_spat = sum(exp.(spec_s .* tau))
        rp1[t] = (rp1_temp * rp1_spat^(3.0))/(L^4)
    end

    return rp1

end

function return_probability_1_per_sum!(rp1, spec_t, spec_s, j::Float64, k::Float64, taus::Vector{Float64}, L::Int64)

    fill!(rp1, 0.0)
    spectrum_Laplace_1per_temp!(spec_t, j, k, L)
    spectrum_Laplace_1per_spat!(spec_s, j, k, L)

    for t in 1:length(taus)
        tau = taus[t]
        rp1_temp, rp1_spat = 0.0, 0.0
        for i = 1:L
            rp1_temp += exp(spec_t[i] * tau)
            rp1_spat += exp(spec_s[i] * tau)
        end
        rp1[t] = (rp1_temp * rp1_spat^(3.0))/(L^4)
    end

    return
end


# 1-periodic Classical return probability #

function return_probability_1_per(jin::Float64,k1::Float64,tau::Float64)

    # 4-volumes

    V = vol_frusta(jin,jin,k1)

    #Weights in temporal direction

    v1 = vol_3d_cube(jin) / dual_len_t(jin,jin,jin,k1,k1)

    #Spatial weights

    w1 = vol_3d_frusta(jin,jin,k1) / dual_len_sp(jin,jin,k1)

    #Compute the return prob in temporal direction

    res1 = vegas( (x,f) -> f[1] = real( exp(-2 * tau * v1/V * (1 - cos(2*pi*x[1] - pi)))),
            1,1,minevals=1e3,maxevals=1e4)

    valres_t = res1[1][1]

    #Compute the return prob in spatial direction

    res2 = vegas( (x,f) -> f[1] = real( exp(-2 * tau * w1/V * (1 - cos(2*pi*x[1] - pi)))),
            1,1,minevals=1e3,maxevals=1e4)

    valres_s = res2[1][1]

    #Compute total return probability

    result = valres_t * (valres_s)^(3.0)

    return result

end

# 1-periodic vectorized  classical return probability #

function rp1vec(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64)

    N_spins = Int(2*(j_max - j_min) + 1)

    spin_tensor = [(0.5*(i - 1) + j_min, 0.5*(j - 1) + j_min) for i in 1:N_spins, j in 1:N_spins]
    spins = vec(spin_tensor)

    tensor = zeros(Float64, N_spins^2, length(tau_range))
    
    ftensor = Array{Future}(undef, size(tensor))

    for t in 1:length(tau_range)
        for s in 1:length(spins)
            i = spins[s][1]
            j = spins[s][2]
            p = workers()[mod1(n, nworkers())]
            ftensor[s,t] = remotecall(return_probability_1_per, p, i, j, tau_range[t]) 
        end      
    end

    for t in 1:length(tau_range)
        for s in 1:length(spins)
            i = spins[s][1]
            j = spins[s][2]
            tensor[s,t] = fetch(ftensor[s,t])
        end
        if count(x -> (x < 0), tensor[:,t]) > 0
            return "Tensor contains negative entries. Need to increase evals!"
        end  
    end

    save("/beegfs/ri47hud/specdim_codes/RP1/rp1.jld", "rp1", tensor)

    return tensor

end

# Implement Riemannian EPRL-condition #

function rp1vec_EPRL(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64)

    N = Int(2*(j_max - j_min) + 1)
    tensor = spzeros(N^2, length(tau_range))
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N for a_0 in A_i, a_1 in A_i.-1])

    ftensor = Array{Future}(undef, length(nzeros), length(tau_range))

    for t in 1:length(tau_range)
        for (ind, nz) in enumerate(nzeros)
        i = 0.5*nconvert1(nz-1, N)[1] + j_min
        j = 0.5*nconvert1(nz-1, N)[2] + j_min
        p = workers()[mod1(ind, nworkers())] 
            ftensor[ind,t] = remotecall(return_probability_1_per, p, i, j, tau_range[t]) 
        end
    end

    TAUs    = Int64[]
    NZs     = Int64[]
    Vs      = Float64[]

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros)
            v = fetch(ftensor[ind,t])
            if v != 0
                push!(TAUs, t)
                push!(NZs, nz)
                push!(Vs, v)
            end
        end
    end

    tensor = sparse(TAUs, NZs, Vs, length(tau_range), N^2)

    return tensor

end

function rp1vec_EPRL_sum(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, L::Int64)

    N = Int(2*(j_max - j_min) + 1)
    #tensor = zeros(length(tau_range), N^2)
    #tensor = [zeros(length(tau_range)) for i in 1:N^2]
    
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N for a_0 in A_i, a_1 in A_i.-1])

    Is = Int64[]
    Js = Int64[]
    Ts = Float64[]
    rp1 = zeros(length(tau_range))
    spec_t = zeros(L)
    spec_s = zeros(L)

    #for n in 1:length(nzeros)
        #s = nzeros[n]
    #for (n, s) in enumerate(nzeros)
    for s in nzeros
        i = 0.5*nconvert1(s-1, N)[1] + j_min
        j = 0.5*nconvert1(s-1, N)[2] + j_min
        return_probability_1_per_sum!(rp1, spec_t, spec_s, i, j, tau_range, L)
        for (ii,Ti) in enumerate(rp1)
            #if isapprox(Ti, 0.0)
            if Ti != 0.0
                push!(Is, ii)
                push!(Js, s)
                push!(Ts, Ti)
            end
        end
    end

    tensor = sparse(Is, Js, Ts)

    return tensor

    #for n in 1:length(nzeros)
        #s = nzeros[n]
        #i = 0.5*nconvert1(s-1, N)[1] + j_min
        #j = 0.5*nconvert1(s-1, N)[2] + j_min
        #vt = view(tensor, :, s)
        #vt .= return_probability_1_per_sum(i, j, tau_range, L)
        #tensor[s] .= return_probability_1_per_sum(i, j, tau_range, L)
    #end

    #return reduce(hcat, tensor)

end

# Restrict a 1-periodic return probability vector #

function nconvert1(n::Int64, Nspins::Int64)

    a_1 = floor(n/Nspins)
    a_0 = float(mod(n,Nspins))

    return (a_0, a_1)

end

function rp1_res(taus::Vector{Float64}, rp1::SparseMatrixCSC{Float64, Int64}, gamma::Float64, jmin::Float64, jmax::Float64, jmin_new::Float64, jmax_new::Float64)

    #Index set of original tensor
    N = Int(2*(jmax - jmin) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(gamma)) == 1, [0.5*(i - 1) + jmin for i in 1:N])

    #Index set of sub tensor
    a_min = 2*(jmin_new - jmin)
    a_max = 2*(jmax_new - jmin)
    A_i_sub = filter(x -> a_min <= x <= a_max+1, A_i)
    nzeros_sub = vec([a_0 + a_1*N for a_0 in A_i_sub, a_1 in A_i_sub.-1])
    #println("The length of nzeros_sub is ", length(nzeros_sub))

    #Index set of new tensor
    N_new = Int(2*(jmax_new - jmin_new) + 1)
    A_i_new = findall(n -> EPRL_cond(n, rationalize(gamma)) == 1, [0.5*(i - 1) + jmin_new for i in 1:N_new])
    nzeros_new = vec([a_0 + a_1*N_new for a_0 in A_i_new, a_1 in A_i_new.-1])
    #println("The number of non-zero configurations in the new tensor is ", length(nzeros_new))

    rp1_new = spzeros(N_new^2, length(taus))
    for t in 1:length(taus)
        for n in 1:length(nzeros_new)
            n_new = nzeros_new[n]
            n_sub = nzeros_sub[n]
            rp1_new[n_new,t] = rp1[n_sub,t]
        end
    end

    return rp1_new

end

######--- 2-periodic ---######

# 2-periodic Laplacian and its spectrum #

function Laplace_2per(jin::Float64,jmid::Float64,k1::Float64,k2::Float64,
    p::Float64,qx::Float64,qy::Float64,qz::Float64)

    mat = complex(zeros(2,2))

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2)

        #Define the weights first

        # 4-volumes

        V1 = vol_frusta(jin,jmid,k1)
        V2 = vol_frusta(jmid,jin,k2)

        #Weights in temporal direction

        v1 = vol_3d_cube(jmid) / dual_len_t(jin,jmid,jin,k1,k2)
        v2 = vol_3d_cube(jin) / dual_len_t(jmid,jin,jmid,k2,k1)

        #Spatial weights

        w1 = vol_3d_frusta(jin,jmid,k1) / dual_len_sp(jin,jmid,k1)
        w2 = vol_3d_frusta(jmid,jin,k2) / dual_len_sp(jmid,jin,k2)

        W1 = 2 * w1 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))
        W2 = 2 * w2 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))

        mat[1,1] = 1/V1 * (v1 + v2 + W1)

        mat[1,2] = -1/V1 * (v1 + v2 * exp(-1im * p))

        mat[2,1] = -1/V2 * (v1 + v2 * exp(1im * p))

        mat[2,2] = 1/V2 * (v1 + v2 + W2)

    end

    return mat

end


function spectrum_Laplace_2per(jin::Float64,jmid::Float64,k1::Float64,k2::Float64,
    p::Float64,qx::Float64,qy::Float64,qz::Float64)

    res_1, res_2 = 0.0, 0.0

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2)

        V1 = vol_frusta(jin,jmid,k1)
        V2 = vol_frusta(jmid,jin,k2)

        v1 = vol_3d_cube(jmid) / dual_len_t(jin,jmid,jin,k1,k2)
        v2 = vol_3d_cube(jin) / dual_len_t(jmid,jin,jmid,k2,k1)

        w1 = vol_3d_frusta(jin,jmid,k1) / dual_len_sp(jin,jmid,k1)
        w2 = vol_3d_frusta(jmid,jin,k2) / dual_len_sp(jmid,jin,k2)

        W1 = 2 * w1 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))
        W2 = 2 * w2 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))

        a = -(V1 * (v1 + v2 + W2) + V2 * (v1 + v2 + W1)) / (V1 * V2)
        b = 1/(V1 * V2) * ((v1 + v2 + W1) * (v1 + v2 + W2) - v1^2 -v2^2 - 2 * v1 * v2 * cos(p))

        res_1 = real(-a/2. - sqrt(complex((a/2.)^2 - b)))
        res_2 = real(-a/2. + sqrt(complex((a/2.)^2 - b)))

    end

    return [res_1, res_2]

end

function spectrum_Laplace_2per_corr(jin::Float64,jmid::Float64,k1::Float64,k2::Float64,
    p::Float64,qx::Float64,qy::Float64,qz::Float64)

    res_1, res_2 = 0.0, 0.0

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2)

        V1 = vol_frusta(jin,jmid,k1)
        V2 = vol_frusta(jmid,jin,k2)

        v1 = vol_3d_cube(jmid) / dual_len_t(jin,jmid,jin,k1,k2)
        v2 = vol_3d_cube(jin) / dual_len_t(jmid,jin,jmid,k2,k1)

        w1 = vol_3d_frusta(jin,jmid,k1) / dual_len_sp_corr(jin,jmid,k1)
        w2 = vol_3d_frusta(jmid,jin,k2) / dual_len_sp_corr(jmid,jin,k2)

        W1 = 2 * w1 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))
        W2 = 2 * w2 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))

        a = -(V1 * (v1 + v2 + W2) + V2 * (v1 + v2 + W1)) / (V1 * V2)
        b = 1/(V1 * V2) * ((v1 + v2 + W1) * (v1 + v2 + W2) - v1^2 -v2^2 - 2 * v1 * v2 * cos(p))

        res_1 = real(-a/2. - sqrt(complex((a/2.)^2 - b)))
        res_2 = real(-a/2. + sqrt(complex((a/2.)^2 - b)))

    end

    return [res_1, res_2]

end

function spectrum_Laplace_2per_disc(jin::Float64,jmid::Float64,k1::Float64,k2::Float64, L::Int64)

    spec_pre = [spectrum_Laplace_2per(jin, jmid, k1, k2, (2*pi/L)*p, (2*pi/L)*qx, (2*pi/L)*qy, (2*pi/L)*qz) for p in 0:(L-1), qx in 0:(L-1), qy in 0:(L-1), qz in 0:(L-1)]
    spec = reduce(hcat,spec_pre)
    
    return spec

end

function spectrum_Laplace_2per_disc!(spec, jin::Float64,jmid::Float64,k1::Float64,k2::Float64, L::Int64)

    for p = 0:L-1
        for qx = 0:L-1
            for qy = 0:L-1
                for qz = 0:L-1
                    spec_temp = spectrum_Laplace_2per(jin, jmid, k1, k2, (2*pi/L)*p, (2*pi/L)*qx, (2*pi/L)*qy, (2*pi/L)*qz)
                    spec[1, p+1, qx + 1, qy + 1, qz + 1] = spec_temp[1]
                    spec[2, p+1, qx + 1, qy + 1, qz + 1] = spec_temp[2]
                end
            end
        end
    end
    
    return

end

# 2-periodic classical return probability #

function return_probability_2_per(jin::Float64,jmid::Float64,k1::Float64,k2::Float64,tau::Float64, emin::Float64, emax::Float64)

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2) && SU2_cond(jin, jmid) == 1

        res = cuhre( (x,f) -> f[1] = real(0.5 * sum(exp.(-tau * spectrum_Laplace_2per(jin,jmid,k1,k2,
        2*pi*x[1] - pi,2*pi*x[2] - pi,2*pi*x[3] - pi,2*pi*x[4] - pi) ))),
            4, 1, minevals=emin, maxevals=emax
        )

        return res[1][1]

    else 

        return 0.0

    end 

end

function return_probability_2_per_corr(jin::Float64,jmid::Float64,k1::Float64,k2::Float64,tau::Float64, emin::Float64, emax::Float64)

    if -1. /sqrt(2) <= (jmid - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - jmid) / (4*k2) <= 1. /sqrt(2) && SU2_cond(jin, jmid) == 1

        res = cuhre( (x,f) -> f[1] = real(0.5 * sum(exp.(-tau * spectrum_Laplace_2per_corr(jin,jmid,k1,k2,
        2*pi*x[1] - pi,2*pi*x[2] - pi,2*pi*x[3] - pi,2*pi*x[4] - pi) ))),
            4, 1, minevals=emin, maxevals=emax
        )

        return res[1][1]

    else 

        return 0.0

    end 

end

function return_probability_2per_sum(jin::Float64, jmid::Float64, k1::Float64, k2::Float64, taus::Vector{Float64}, L::Int64)

    spec = spectrum_Laplace_2per_disc(jin, jmid, k1, k2, L)
    rp2 = zeros(Float64, length(taus))

    for t in 1:length(taus)
        tau = taus[t]
        rp2[t] = (sum(exp.(-spec[1,:] .* tau)) + sum(exp.(-spec[2,:] .* tau)))/(2 * L^4) 
    end

    return rp2
    
end

function return_probability_2per_sum!(rp2, spec, jin::Float64, jmid::Float64, k1::Float64, k2::Float64, taus::Vector{Float64}, L::Int64)

    fill!(rp2, 0.0)
    spectrum_Laplace_2per_disc!(spec, jin, jmid, k1, k2, L)

    for (t, tau) in enumerate(taus) 
        rp2[t] = (sum(exp.(-spec[1,:,:,:,:] .* tau)) + sum(exp.(-spec[2,:,:,:,:] .* tau)))/(2 * L^4) 
    end

    return rp2
    
end

# 2-periodic vectorized and distributed classical return probability

# Implement Riemannian EPRL-condition #

function rp2vec_EPRL_sum(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, L::Int64)

    N = Int(2*(j_max - j_min) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])
    
    ftensor = Array{Future}(undef, length(nzeros))


    for (ind,nz) in enumerate(nzeros)
        rp2 = zeros(length(tau_range))
        spec = zeros(2, L, L, L, L)
        i = 0.5*nconvert2(nz-1, N)[1] + j_min
        j = 0.5*nconvert2(nz-1, N)[2] + j_min
        k = 0.5*nconvert2(nz-1, N)[3] + j_min
        l = 0.5*nconvert2(nz-1, N)[4] + j_min
        p = workers()[mod1(ind, nworkers())]
        ftensor[ind] = remotecall(return_probability_2per_sum!, p, rp2, spec, i, j, k, l, tau_range, L) 
    end

    TAUs = Int64[]
    NZs   = Int64[]
    Vs   = Float64[]

    for (ind, nz) in enumerate(nzeros)
        temp = fetch(ftensor)
        for (tau, v) in enumerate(fetch(ftensor[ind]))
            if v != 0
                push!(TAUs, tau)
                push!(NZs, nz)
                push!(Vs, v)
            end
        end
    end

    tensor = sparse(TAUs, NZs, Vs, length(tau_range), N^4)

    return tensor

end

function rp2vec_EPRL_sum_partial(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, L::Int64, N_parts::Int64, part::Int64)

    N = Int(2*(j_max - j_min) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])
    
    ind_min = Int((length(nzeros)/N_parts)*(part - 1) + 1)
    ind_max = Int((length(nzeros)/N_parts)*part)
    nzeros_res = nzeros[ind_min:ind_max]
    
    ftensor = Array{Future}(undef, length(nzeros_res))

    for (ind, nz) in enumerate(nzeros_res)
        rp2 = zeros(length(tau_range))
        spec = zeros(2, L, L, L, L)
        i = 0.5*nconvert2(nz-1, N)[1] + j_min
        j = 0.5*nconvert2(nz-1, N)[2] + j_min
        k = 0.5*nconvert2(nz-1, N)[3] + j_min
        l = 0.5*nconvert2(nz-1, N)[4] + j_min
        p = workers()[mod1(ind, nworkers())]
        ftensor[ind] = remotecall(return_probability_2per_sum!, p, rp2, spec, i, j, k, l, tau_range, L) 
    end

    TAUs = Int64[]
    NZs  = Int64[]
    Vs   = Float64[]

    for (i,nz) in enumerate(nzeros_res)
        for (tau, v) in enumerate(fetch(ftensor[i]))
            if v != 0
                push!(TAUs, tau)
                push!(NZs, nz)
                push!(Vs, v)
            end
        end
    end

    tensor = sparse(TAUs, NZs, Vs, length(tau_range), N^4)

    return tensor

end

function rp2vec_EPRL_int(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, emin::Float64, emax::Float64)

    N = Int(2*(j_max - j_min) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])

    ftensor = Array{Future}(undef, length(nzeros), length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros)
            i = 0.5*nconvert2(nz-1, N)[1] + j_min
            j = 0.5*nconvert2(nz-1, N)[2] + j_min
            k = 0.5*nconvert2(nz-1, N)[3] + j_min
            l = 0.5*nconvert2(nz-1, N)[4] + j_min
            p = workers()[mod1(ind, nworkers())]
            ftensor[ind,t] = remotecall(return_probability_2_per, p, i, j, k, l, tau_range[t], emin, emax) 
        end
    end

    tensor = spzeros(N^4, length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros)
            tensor[nz,t] = fetch(ftensor[ind,t])
        end
    end

    return tensor

end

function rp2vec_EPRL_int_corr(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, emin::Float64, emax::Float64)

    N = Int(2*(j_max - j_min) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])

    ftensor = Array{Future}(undef, length(nzeros), length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros)
            i = 0.5*nconvert2(nz-1, N)[1] + j_min
            j = 0.5*nconvert2(nz-1, N)[2] + j_min
            k = 0.5*nconvert2(nz-1, N)[3] + j_min
            l = 0.5*nconvert2(nz-1, N)[4] + j_min
            p = workers()[mod1(ind, nworkers())]
            ftensor[ind,t] = remotecall(return_probability_2_per_corr, p, i, j, k, l, tau_range[t], emin, emax) 
        end
    end

    tensor = spzeros(N^4, length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros)
            tensor[nz,t] = fetch(ftensor[ind,t])
        end
    end

    return tensor

end


function rp2vec_EPRL_int_partial(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, emin::Float64, emax::Float64, N_parts::Int64, part::Int64)

    N = Int(2*(j_max - j_min) + 1)
    A_i = findall(x -> EPRL_cond(x, rationalize(γ)) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1])

    ind_min = Int((length(nzeros)/N_parts)*(part - 1) + 1)
    ind_max = Int((length(nzeros)/N_parts)*part)
    nzeros_res = nzeros[ind_min:ind_max]

    ftensor = Array{Future}(undef, length(nzeros_res), length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros_res)
            i = 0.5*nconvert2(nz-1, N)[1] + j_min
            j = 0.5*nconvert2(nz-1, N)[2] + j_min
            k = 0.5*nconvert2(nz-1, N)[3] + j_min
            l = 0.5*nconvert2(nz-1, N)[4] + j_min
            p = workers()[mod1(ind, nworkers())]
            ftensor[ind,t] = remotecall(return_probability_2_per, p, i, j, k, l, tau_range[t], emin, emax) 
        end
    end

    tensor = spzeros(N^4, length(tau_range))

    for t in 1:length(tau_range)
        for (ind,nz) in enumerate(nzeros_res)
            tensor[nz,t] = fetch(ftensor[ind,t])
        end
    end

    return tensor

end

# Restrict a 2-periodic return probability vector #

function nconvert2(n::Int64, Nspins::Int64)

    a_3 = floor(n/Nspins^3)
    a_2 = floor(mod(n, Nspins^3)/Nspins^2)
    a_1 = floor(mod(n, Nspins^2)/Nspins^1)
    a_0 = float(mod(n, Nspins))

    return (a_0, a_1, a_2, a_3)

end

function rp2_res(rp2::Array{Float64, 2}, j_min::Float64, j_max::Float64, j_min_new::Float64, j_max_new::Float64)

    len_tau = length(rp2[1,:])
    Nspins = Int(2*(j_max - j_min) + 1)
    a_min = 2*(j_min_new - j_min)
    a_max = 2*(j_max_new - j_min)
    ind_res = [i for i in 1:Nspins^4 if (nconvert2(i-1,Nspins) .>= a_min) == (1 ,1, 1, 1) && (nconvert2(i-1,Nspins) .<= a_max) == (1, 1, 1, 1)]
    rp2_res = [rp2[ind_res[i],t] for i in 1:length(ind_res), t in 1:len_tau]

    return rp2_res

end

######--- 3-periodic ---######

# 3-periodic Laplacian #

function Laplace_3per(jin::Float64, j1::Float64, j2::Float64, k1::Float64, k2::Float64, k3::Float64,
    p::Float64,qx::Float64,qy::Float64,qz::Float64)

    mat = complex(zeros(3,3))

    if -1. /sqrt(2) <= (j1 - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (j2 - j1) / (4*k2) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - j2) / (4*k3) <= 1. /sqrt(2)


        #Define the weights first

        # 4-volumes

        V1 = vol_frusta(jin,j1,k1)
        V2 = vol_frusta(j1,j2,k2)
        V3 = vol_frusta(j2,jin,k3)

        #Weights in temporal direction

        v1 = vol_3d_cube(j1) / dual_len_t(jin,j1,j2,k1,k2)
        v2 = vol_3d_cube(j2) / dual_len_t(j1,j2,jin,k2,k3)
        v3 = vol_3d_cube(jin) / dual_len_t(j2,jin,j1,k3,k1)

        #Spatial weights

        w1 = vol_3d_frusta(jin,j1,k1) / dual_len_sp(jin,j1,k1)
        w2 = vol_3d_frusta(j1,j2,k2) / dual_len_sp(j1,j2,k2)
        w3 = vol_3d_frusta(j2,jin,k3) / dual_len_sp(j2,jin,k3)

        W1 = 2 * w1 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))
        W2 = 2 * w2 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))
        W3 = 2 * w3 * ((1 - cos(qx)) + (1 - cos(qy)) + (1 - cos(qz)))

        mat[1,1] = 1/V1 * (v3 + v1 + W1)

        mat[1,2] = -1/V1 * v1 

        mat[1,3] = -1/V1 * v3 * exp(- 1im * p)

        mat[2,1] = -1/V2 * v1 

        mat[2,2] = 1/V2 * (v1 + v2 + W2)

        mat[2,3] = -1/V2 * v2  

        mat[3,1] = -1/V3 * v3 * exp(1im * p)

        mat[3,2] = -1/V3 * v2  

        mat[3,3] = 1/V3 * (v2 + v3 + W3)

    end

    return mat

end

# 3-periodic classical return probability #

function return_probability_3_per(jin::Float64, j1::Float64, j2::Float64,
    k1::Float64, k2::Float64, k3::Float64,tau::Float64, emin::Float64, emax::Float64)

    if -1. /sqrt(2) <= (j1 - jin) / (4*k1) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (j2 - j1) / (4*k2) <= 1. /sqrt(2) &&
        -1. /sqrt(2) <= (jin - j2) / (4*k3) <= 1. /sqrt(2) &&
        SU2_cond(jin, j1) == 1 && SU2_cond(j1, j2) == 1 &&
        SU2_cond(j2, jin) == 1

        res = cuhre( (x,f) -> f[1] = real((1/3.) * sum(
        exp.(-tau * eigvals(Laplace_3per(jin, j1, j2, k1, k2, k3, 2*pi*x[1] - pi, 2*pi*x[2] - pi, 2*pi*x[3] - pi, 2*pi*x[4] - pi)) )
        )),4,1,minevals=emin, maxevals=emax)

        return res[1][1]

    else

        return 0.0

    end

end

# 3-periodic vectorized and distributed classical return probability #

function rp3vec(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, emin::Float64, emax::Float64)

    N_spins = Int(2*(j_max - j_min) + 1)

    spin_tensor = [(0.5*(i - 1) + j_min, 0.5*(j - 1) + j_min, 0.5*(k - 1) + j_min, 0.5*(l - 1) + j_min, 0.5*(m - 1) + j_min, 0.5*(n - 1) + j_min )
     for i in 1:N_spins, j in 1:N_spins, k in 1:N_spins, l in 1:N_spins, m in 1:N_spins, n in 1:N_spins]
    spins = vec(spin_tensor)

    tensor = zeros(Float64, N_spins^6, length(tau_range))
    
    ftensor = Array{Future}(undef, size(tensor))

    for t in 1:length(tau_range)
        for s in 1:length(spins)
            i = spins[s][1]
            j = spins[s][2]
            k = spins[s][3]
            l = spins[s][4]
            m = spins[s][5]
            n = spins[s][6]
            p = workers()[mod1(n, nworkers())]
            ftensor[s,t] = remotecall(return_probability_3_per, p, i, j, k, l, m, n, tau_range[t], emin, emax) 
        end      
    end

    for t in 1:length(tau_range)
        for s in 1:length(spins)
            i = spins[s][1]
            j = spins[s][2]
            k = spins[s][3]
            l = spins[s][4]
            m = spins[s][5]
            n = spins[s][6]
            tensor[s,t] = fetch(ftensor[s,t])
        end
        if count(x -> (x < 0), tensor[:,t]) > 0
            return "Tensor contains negative entries. Need to increase evals!"
        end  
    end

    #save("RP3_vecs/rp3vec.jld", "rp3vec", tensor)

    return tensor

end

# Implement Riemannian EPRL-condition #


function rp3vec_EPRL(tau_range::Vector{Float64}, j_min::Float64, j_max::Float64, γ::Float64, emin::Float64, emax::Float64)

    N = Int(2*(j_max - j_min) + 1)
    tensor = spzeros(N^6, length(tau_range))
    A_i = findall(x -> EPRL_cond(x, γ) == 1, [0.5*(i - 1) + j_min for i in 1:N])
    nzeros = vec([a_0 + a_1*N + a_2*N^2 + a_3*N^3 + a_4*N^4 + a_5*N^5 
    for a_0 in A_i, a_1 in A_i.-1, a_2 in A_i.-1, a_3 in A_i.-1, a_4 in A_i.-1, a_5 in A_i.-1])

    ftensor = Array{Future}(undef, length(nzeros), length(tau_range))

    for t in 1:length(tau_range)
        for n in 1:length(nzeros)
            s = nzeros[n]
            i = 0.5*nconvert3(s-1, N)[1] + j_min
            j = 0.5*nconvert3(s-1, N)[2] + j_min
            k = 0.5*nconvert3(s-1, N)[3] + j_min
            l = 0.5*nconvert3(s-1, N)[4] + j_min
            m = 0.5*nconvert3(s-1, N)[5] + j_min
            q = 0.5*nconvert3(s-1, N)[6] + j_min
            p = workers()[mod1(n, nworkers())]
            ftensor[n,t] = remotecall(return_probability_3_per, p, i, j, k, l, m, q, tau_range[t], emin, emax) 
        end
    end

    for t in 1:length(tau_range)
        for n in 1:length(nzeros)
            s = nzeros[n]
            tensor[s,t] = fetch(ftensor[n,t])
        end
    end

    return tensor

end



# Restrict a 3-periodic return probability vector #

function nconvert3(n::Int64, Nspins::Int64)

    a_5 = floor(n/Nspins^5)
    a_4 = floor(mod(n, Nspins^5)/Nspins^4)
    a_3 = floor(mod(n, Nspins^4)/Nspins^3)
    a_2 = floor(mod(n, Nspins^3)/Nspins^2)
    a_1 = floor(mod(n, Nspins^2)/Nspins^1)
    a_0 = float(mod(n, Nspins))

    return (a_0, a_1, a_2, a_3, a_4, a_5)

end

function rp3_res(rp3::Array{Float64, 2}, j_min::Float64, j_max::Float64, j_min_new::Float64, j_max_new::Float64)

    len_tau = length(rp3[1,:])
    Nspins = Int(2*(j_max - j_min) + 1)
    a_min = 2*(j_min_new - j_min)
    a_max = 2*(j_max_new - j_min)
    ind_res = [i for i in 1:Nspins^6 if (nconvert3(i-1,Nspins) .>= a_min) == (1 ,1, 1, 1, 1, 1) 
    && (nconvert3(i-1,Nspins) .<= a_max) == (1, 1, 1, 1, 1, 1)]
    rp3_res = [rp3[ind_res[i],t] for i in 1:length(ind_res), t in 1:len_tau]

    return rp3_res

end