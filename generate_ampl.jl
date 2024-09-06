using JLD, LinearAlgebra, Cuba, SparseArrays
# Codes for the semi-classical amplitudes
include("/home/ri47hud/codes/Spectral Dimension/basic_frusta.jl")
include("/home/ri47hud/codes/Spectral Dimension/laplacians_v3.jl")
include("/home/ri47hud/codes/Spectral Dimension/vertex_ampl.jl")
include("/home/ri47hud/codes/Spectral Dimension/return_prob_v2.jl")

alph = [i/144 for i in 102:105]
gamma =     1/3
G =         1.0
Lambda =    0.0
Ampl2 = [Am16_vector(a, G, gamma, Lambda, 0.5, 100.0) for a in alph]
save("/home/ri47hud/codes/Spectral Dimension/Ampl2.jld", "Ampl2", Ampl2)
