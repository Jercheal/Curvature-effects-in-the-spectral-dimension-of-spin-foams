function cosphi(j1::Float64,j2::Float64,j3::Float64)

    res = (j2-j1)/(4*j3)

    res = real(res)

    res


end

function K(j1::Float64,j2::Float64,j3::Float64)

    res = sqrt(1 - 2 * ((complex(j1)-complex(j2))/(4 * complex(j3)))^2)

    res = real(res)

    res

end

function D(j1::Float64,j2::Float64,j3::Float64)

    res = 16 * j1^3 * j2^3 * j3^15 * (-1 + 2*cosphi(j1,j2,j3)^2 - 1im * K(j1,j2,j3) )
    res *= (-2 + cosphi(j1,j2,j3)^2 + 1im * K(j1,j2,j3))^2
    res *= (1 + cosphi(j1,j2,j3)^2 + (j1+j2)/(2*j3))^3
    res *= (1 + 2*cosphi(j1,j2,j3)^2 + (j1+j2)/(2*j3) - 1im * K(j1,j2,j3))^3
    res *= (1 + cosphi(j1,j2,j3)^2 + (1-cosphi(j1,j2,j3)^2) * (j1+j2)/j3 + 1im *
    K(j1,j2,j3)*(1- 3* cosphi(j1,j2,j3)^2))^3

    res
end


#----------------------------------------------


### VOLUME AND HEIGHT OF HYPERFRUSTUM

function vol_frusta(jin::Float64,jfin::Float64,k::Float64)

    res = 1/2. * k * (jin + jfin) * sqrt(complex(1 - ((jfin - jin)^2.) / (8 * k^2.)))

    res = real(res)

    res

end


function height_frusta(jin::Float64,jfin::Float64,k::Float64)

    res = (2*k)/(sqrt(jfin) + sqrt(jin)) * sqrt(complex(1 - (jfin - jin)^2 / (8 * k^2)))

    res = real(res)

    res

end

### sideface gives the spin of the side face as a function of top, bottom spins and height of hyperfrustum

function sideface(jin::Float64,jfin::Float64,H::Float64)

    res = sqrt(((sqrt(jin)+sqrt(jfin))/2*H)^2 + (jfin - jin)^2/8)

    res = real(res)

    res

end

function vol_3d_cube(j::Float64)

    res = j^(3/2)

    res

end

#### 3-volume of frustum

function vol_3d_frusta(jin::Float64,jfin::Float64,k::Float64)

    res = 1/3. * sqrt( complex((2 * k / (sqrt(jin) + sqrt(jfin)))^2 - ( 1/2 * (sqrt(jin) - sqrt(jfin)) )^2 )) *
     (jin + sqrt(jin * jfin) + jfin)

    real(res)

end


#### Dual lengths

# Dual length: spatial

function dual_len_sp(jin::Float64,jfin::Float64,k::Float64)

    if -1. /sqrt(2) <= (jfin - jin) / (4*k) <= 1. /sqrt(2)

        angle = acos(cosphi(jfin,jin,k)^2 / (1 - cosphi(jfin,jin,k)^2))

        res = 1/2. * (sqrt(jin) + sqrt(jfin)) * cos(pi/2 - angle)

    else

        res = 1/2. * (sqrt(jin) + sqrt(jfin))

    end

    res

end

function dual_len_sp_corr(jin::Float64,jfin::Float64,k::Float64)

    if -1. /sqrt(2) <= (jfin - jin) / (4*k) <= 1. /sqrt(2)

        angle = acos(sqrt(cosphi(jfin,jin,k)^2 / (1 - cosphi(jfin,jin,k)^2)))

        res = 1/2. * (sqrt(jin) + sqrt(jfin)) * cos(pi/2 - angle)

    else

        res = 1/2. * (sqrt(jin) + sqrt(jfin))

    end

    res

end

# Dual length temporal (just the average of heights in slices n and n+1)

function dual_len_t(jin::Float64,jmid::Float64,jfin::Float64,k1::Float64,k2::Float64)

    res = 1/2. * (height_frusta(jin,jmid,k1) + height_frusta(jmid,jfin,k2))

    res

end


#----------------------------------------------


### REGGE ACTION AND CRITICAL ANGLE θ

function costheta(j1::Float64,j2::Float64,j3::Float64)

    res = cosphi(j1,j2,j3) / sqrt(complex(1-cosphi(j1,j2,j3)^2))

    res = real(res)

    res

end

function SRegge(j1::Float64,j2::Float64,j3::Float64)

    res = 6* (j1-j2) * (pi/2 - acos(costheta(j1,j2,j3))) + 12 * j3 * (pi/2 - acos(costheta(j1,j2,j3)^2))

    res = real(res)

    res

end




#----------------------------------------------




### VERTEX, EDGE AND FACE AMPLITUDES

function VAmp(G::Float64,γ::Float64,Lambda::Float64,j1::Float64,j2::Float64,j3::Float64)

    res = 0

    if -1. /sqrt(2) <= (j2 - j1) / (4*j3) <= 1. /sqrt(2)

        res = exp(1im *(1/(G))*SRegge(j1,j2,j3))/(-D(j1,j2,j3))
        res += exp(-1im *(1/(G))*SRegge(j1,j2,j3))/(-conj(D(j1,j2,j3)))
        res += 2*cos((1/(G)) * γ * SRegge(j1,j2,j3) - (1/(G)) * Lambda * vol_frusta(j1,j2,j3))/abs(D(j1,j2,j3))

        #res *= 1/(pi)^7 * (1./ sqrt(1 - γ^2))^21

    end

    #res = real(res)

    res

end



function EAmp(γ::Float64,j1::Float64,j2::Float64,j3::Float64)

    phi = acos((j2-j1)/ (4*j3))
   
    det = j3 / 2 * sin(phi)^2 * (j1 + j2 + 2*j3 * (1 + cos(phi)^2))^2

    #res = j3 * (1-cosphi(j1,j2,j3)^2) * (j1 + j2 + 2*j3 * (1 + cosphi(j1,j2,j3)^2))^2

    #res *= 1/(4*pi)^4 * (sqrt(1 - γ^2) / (8*pi))^3

    #res = real(res)

    return det

end



function FAmp(alpha::Float64,γ::Float64,j::Float64)

    #res = (j^2*(1-γ^2))^α
    res = (j^2)^alpha

    #res = real(res)

    res

end

### Semiclassical SU2-amplitudes

function EAmpSU2(j1::Float64,j2::Float64,j3::Float64)
    
    phi = acos((j2-j1)/ (4*j3))
   
    det = j3 / 2 * sin(phi)^2 * (j1 + j2 + 2*j3 * (1 + cos(phi)^2))^2
    
    res = ((4*pi)^2 / (2 * (2*pi)^(3/2))) *  sqrt(det)

    return res
end

function VAmpSU2(j1::Float64,j2::Float64,j3::Float64)

    res = 0.

    if -1. /sqrt(2) <= (j2 - j1) / (4*j3) <= 1. /sqrt(2) #&& iseven(convert(Int,2*(j1 + j2 + 4 * j3)))

        res += exp(1im * SRegge(j1,j2,j3))/(sqrt(-D(j1,j2,j3)))
        res += exp(-1im * SRegge(j1,j2,j3))/(sqrt(-conj(D(j1,j2,j3))))
        
        #res *= 1/(pi)^7 * (1./ sqrt(1 - γ^2))^21

    end

    res = real(res*(2 * pi)^(21/2) * 2^7 / (4 * pi)^14)

    return res

end