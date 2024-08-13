module PIMC

include("clusterization.jl")
include("sys_operators.jl")

using .SysOperators
using .Clusterization
import LinearAlgebra as la
using StatsBase
using Statistics
export PathIntegral, runMC_cluster

################################################################################
@kwdef mutable struct PathIntegral
    ###################################################################################################
	N::Int64
    m_max::Int64
    g::Float64	
    L::Int64
    MC_steps::Int64
    T::Float64 = 0.1
    PIGS::Bool = true
    PBC::Bool = false
    Nskip::Int64 = 1
    Nequilibrate::Int64 = 100
    Nstates::Int64 = 2*m_max+1
    beta::Float64 = 1.0/T
    tau::Float64= beta/L 
    
    # Number of extra beads
	P::Int64 = (sign(N-2)+1)*L+1    
	
	# 	Parameters
	# 	--------------
	# 	N_rotors : int
	# 		The number of rotors to be simulated.
	# 	m_max : int
	# 		The maximum size of the free rotor eigenstate of magnetic moment projection basis.
	# 	g : float
	# 		Non-dimentional interaction strength between the rotors.
	# 	L : int
	# 		The number of original beads to use in the path-integral.
    #   PIGS : bool, optional
    #       Enables path-integral ground state calculations. The default is True.
    #   PBC : bool, optional
    #       Enables periodic boundary conditions in the system chain. The default is False.
	# 	MC_steps : int
	# 		The number of steps to use in the Monte Carlo method.
	# 	Nskip : int, optional
	# 		The number of steps to skip when saving the trajectory. The default is 100.
	# 	Nequilibrate : int, optional
	# 		The number of steps to skip before the average properties are accumulated
	# 		to allow for system equilibration. The default is 0.
	# 	T : float, optional
	# 		The system temperature. The default is 1.
    # 	Returns
    # 	-------------
    # 	All inputs mentioned above are stored as attributes in the system.
    # 	beta: float
    # 		The beta value based on the system temperature.
    # 	tau: float
    # 		The tau value for the path integral method based on the beta value
    # 		and the number of beads.
    #   Finish description later

    K_arr::Vector{Float64} = zeros(Float64, MC_steps)
    K_MC::Float64 = 0.0 
    K_stdError_MC::Float64 = 0.0

    E_arr::Vector{Float64} = zeros(Float64, MC_steps)
    E_MC::Float64 = 0.0
    E_stdError_MC::Float64 = 0.0

    histo_total::Vector{Float64} = zeros(Int64, Nstates)
    histo_initial::Vector{Float64} = zeros(Int64, Nstates)
	
    m_corr::Vector{Float64} = zeros(Float64, MC_steps)

    ergodic::Bool = true
    
	# MC Path
    m0 = m_max + 1 # index corresponding to the m = 0 rotor state in Julia
	# initializing the path with all the rotors in the m = 0 state
	MCpath::Array{Int64,2} = ones(Int64, (P, N)) * m0
    
    # defining the SysOperators

    Vij::Array{Float64,2} = zeros(Float64, (Nstates^2,Nstates^2))
    expVij::Array{Float64,2} = zeros(Float64, (Nstates^2,Nstates^2))

    # initializing the 4-body cluster vectors
    cluster_Vij_vec::Vector{Float64} = zeros(Float64, Nstates^4)
    # cluster_H1j_vec::Vector{Float64} = zeros(Float64, Nstates^4)
    # cluster_HiN_vec::Vector{Float64} = zeros(Float64, Nstates^4)

    cluster_expVij_vec::Vector{Float64} = zeros(Float64, Nstates^4)
    # cluster_expH1j_vec::Vector{Float64} = zeros(Float64, Nstates^4)
    # cluster_expHiN_vec::Vector{Float64} = zeros(Float64, Nstates^4)
end


function runMC_cluster(PI::PathIntegral, initial_state::String = "random", sampling::String = "random")
    ################################################################################################
    #	Performs the monte carlo integration to simulate the system.
    #	
    #	Parameters
    #	----------
    #	initialState : string, optional
    #		Selects the distribution of the initial state of the system. The allowed options are 
	#       random or ordered. The default is random.
    #	sampling : string, optional
    #		Selects the sampling method for the Gibbs sampling. The allowed options are sequential
    #       or random. The default is random
    ################################################################################################	
    
    if (PI.L <= 1)
        error("A minimum of 2 beads must be used")
    end

    # building the cluster vector
    build_cluster_vectors(PI)

	# initializing the MC Path
	# init_MCpath_PIMC(PI, initial_state)
    # display_path(PI.m_max, PI.MCpath, true)

    # println("Start of MC loop\n")
    P,N = PI.P , PI.N
    # Starting the MC steps
    if (sampling == "sequential")
        for mc = 1:PI.MC_steps
            for p = 1:P-1
                for n = 2-(p%2) : 2 : N-1
                    cluster_Gibbs_samp_update(PI,[p,n])
                end
            end
            # end of MC
            if (mc >= PI.Nequilibrate)
                # Kinetic Energy Estimator

                K = 0
                # println(path[P_mid,:])
                for i = 1:PI.N
                    K += (path[P_mid, i] - PI.m_max - 1)^2
                end
                PI.K_arr[mc] = K

                # if (PI.N == 2)			
                #     e0 = mel(PI.MCpath[PI.P-1, 1:2], PI.expVij * PI.Vij, PI.MCpath[PI.P, 1:2])
                #     z0 = mel(PI.MCpath[PI.P-1, 1:2], PI.expVij, PI.MCpath[PI.P, 1:2])			
                #     E0 = e0 / z0
                # else
                #     E0 = 0
                #     num_total_states = PI.Nstates^PI.N
                #     for index = 1:num_total_states
                #         Malpha = map_index2vector(PI.Nstates, PI.N, index)
                #         rho = 1
                #         gamma = 0
                #         for i=1:PI.N-1
                #             if (i%2==0)
                #                 rho *= (mel(PI.MCpath[PI.P-1, i:i+1], PI.expVij, Malpha[i:i+1])
                #                     /
                #                     mel(PI.MCpath[PI.P-1, i:i+1], PI.expVij, PI.MCpath[PI.P, i:i+1])
                #                 )
                #             end
                #             gamma += mel(Malpha[i:i+1], PI.Vij, PI.MCpath[PI.P, i:i+1])
                #         end
                #         # defining gamma
                #         E0 += rho*gamma				
                #     end
                # end

                # Ergodicity counter: FIX IT LATER!!!!
                # if (E0 > -1) && (E0 < 0)
                #     counter_E+=E0
                #     counter+=1

                # # elseif (E0 < -4)
                # #     println("E0=", E0, "   ", PI.MCpath[PI.P-1, 1:2])
                # end

                PI.E_arr[mc] = E0
                PI.K_arr[mc] = E0
            end
        end

        # Ergodicity counter: FIX IT LATER!!!!	
        # if (ergo_count < ergo_tolerance)
        #     PI.ergodic = true
        # end
        # println(counter_E/counter)
        # meanE, stdErrE = calculateError_byBinning(PI.E_arr[PI.Nequilibrate+1:PI.MC_steps])
        # PI.E_MC = meanE
        # PI.E_stdError_MC = stdErrE

        meanK, stdErrK = calculateError_byBinning(PI.K_arr[PI.Nequilibrate+1:PI.MC_steps])
        PI.K_MC = meanK
        PI.K_stdError_MC = stdErrK

    # Random Sampling
    elseif (sampling == "random")
        n_random_steps = (N-1)*PI.L
        for mc = 1:PI.MC_steps
            for mc_i = 1:n_random_steps
                p = rand(1:P-1)
                n = rand(2-(p%2) : 2 : N-1)
                cluster_Gibbs_samp_update(PI,[p,n])
            end
            # end of MC
            if (mc >= PI.Nequilibrate)
                if (PI.N == 2)			
                    e0 = mel(PI.MCpath[PI.P-1, 1:2], PI.expVij * PI.Vij, PI.MCpath[PI.P, 1:2])
                    z0 = mel(PI.MCpath[PI.P-1, 1:2], PI.expVij, PI.MCpath[PI.P, 1:2])			
                    E0 = e0 / z0
                else
                    E0 = 0
                    num_total_states = PI.Nstates^PI.N
                    for index = 1:num_total_states
                        Malpha = map_index2vector(PI.Nstates, PI.N, index)                        
                        rho = 1
                        gamma = 0
                        for i=1:PI.N-1
							remain_Malpha = append!(Malpha[1:i-1],Malpha[i+2:PI.N])
							remain_path = append!(PI.MCpath[PI.P,1:i-1],PI.MCpath[PI.P,i+2:PI.N])

							if remain_Malpha == remain_path
								if (i%2==0)
									rho *= (mel(PI.MCpath[PI.P-1, i:i+1], PI.expVij, Malpha[i:i+1])
									/
									mel(PI.MCpath[PI.P-1, i:i+1], PI.expVij, PI.MCpath[PI.P, i:i+1])
								)
								end
								gamma += mel(Malpha[i:i+1], PI.Vij, PI.MCpath[PI.P, i:i+1]) 								                          	                            
							end
						end
                        rho = 1
                        E0 += rho*gamma				
                    end


                    # for n = 1:PI.N-1
                    #     if n%2 == 0 # n even
                    #         e0 = mel(PI.MCpath[PI.P-1, n:n+1], PI.expHij * PI.Hij, PI.MCpath[PI.P, n:n+1])
                    #         z0 = mel(PI.MCpath[PI.P-1, n:n+1], PI.expHij, PI.MCpath[PI.P, n:n+1])
                    #         E0 += e0 / z0
                    #     else # n odd
                    #         aux = 1
                    #         num_total_states = PI.Nstates^PI.N
                    #         for index = 1:num_total_states
                    #             Malpha = map_index2vector(PI.Nstates, PI.N, index)
                    #             # defining rho
                    #             rho = 1
                    #             for i=2:2:PI.N-1
                    #                 rho *= (mel(PI.MCpath[PI.P-1, i:i+1], PI.expHij, Malpha[i:i+1])
                    #                     /
                    #                     mel(PI.MCpath[PI.P-1, i:i+1], PI.expHij, PI.MCpath[PI.P, i:i+1])
                    #                 )
                    #             end
                    #             # defining gamma
                    #             gamma = mel(Malpha[n:n+1], PI.Hij, PI.MCpath[PI.P, n:n+1])
                    #             aux *= rho*gamma				
                    #         end
                    #         E0 += aux
                    #     end
                    # end
                end

                # Ergodicity counter: FIX IT LATER!!!!
                # if (E0 > -1) && (E0 < 0)
                #     counter_E+=E0
                #     counter+=1

                # # elseif (E0 < -4)
                # #     println("E0=", E0, "   ", PI.MCpath[PI.P-1, 1:2])
                # end

                PI.E_arr[mc] = E0
            end
        end

        # Ergodicity counter: FIX IT LATER!!!!	
        # if (ergo_count < ergo_tolerance)
        #     PI.ergodic = true
        # end
        # println(counter_E/counter)
        meanE, stdErrE = calculateError_byBinning(PI.E_arr[PI.Nequilibrate+1:PI.MC_steps])
        PI.E_MC = meanE
        PI.E_stdError_MC = stdErrE
    end
    # println("End of MC loop\n")
    # display_path(PI.m_max, PI.MCpath, true)
end

function build_cluster_vectors(PI::PathIntegral)
    Nstates = PI.Nstates
    # defining the SysOperators
    ops = RotorOps(m_max=PI.m_max, g=PI.g)
    # Vij = ops.Vij
    # Kij = ops.Kij
    # K1j = ops.K1j
    # KiN = ops.KiN

    PI.Vij = ops.Vij
    PI.expVij = abs.(round.(exp(-PI.tau * PI.Vij), digits=14))

    for mi = 1:PI.Nstates
        for mj = 1:PI.Nstates
            for mk = 1:PI.Nstates
                for ml = 1:PI.Nstates
                    mel_Vij = mel([mi,mj], PI.Vij, [mk,ml])                        
                    mel_expVij = mel([mi,mj], PI.expVij, [mk,ml])

                    idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])

                    PI.cluster_Vij_vec[idx] = mel_Vij
                    PI.cluster_expVij_vec[idx] = mel_expVij
                end
            end
        end
    end
end

# function build_cluster_vectors(PI::PathIntegral)
#     Nstates = PI.Nstates

#     # defining the SysOperators
#     ops = RotorOps(m_max=m_max, g=g)
#     Vij = ops.Vij
#     Kij = ops.Kij
#     K1j = ops.K1j
#     KiN = ops.KiN

#     Hij = Vij + Kij
#     expHij = abs.(round.(exp(-tau * Hij), digits=14))

#     if (PI.N>2)
#         H1j = Vij + K1j
#         HiN = Vij + KiN
#         expH1j = abs.(round.(exp(-tau * H1j), digits=14))
#         expHiN = abs.(round.(exp(-tau * HiN), digits=14))
#     end

#     for mi = 1:Nstates
#         for mj = 1:Nstates
#             for mk = 1:Nstates
#                 for mi = 1:Nstates
#                     for ml = 1:Nstates
#                         mel_Hij = mel([mi,mj], Hij, [mk,ml])                        
#                         mel_expHij = mel([mi,mj], expHij, [mk,ml])

#                         idx = map_vector2index(Nstates, [mi,mj,mk,ml])

#                         PI.cluster_Hij_vec[idx] = mel_Hij
#                         PI.cluster_expHij_vec[idx] = mel_expHij

#                         if (PI.N>2)
#                             mel_H1j = mel([mi,mj], H1j, [mk,ml])
#                             mel_HiN = mel([mi,mj], HiN, [mk,ml])
#                             mel_expH1j = mel([mi,mj], expH1j, [mk,ml])
#                             mel_expHiN = mel([mi,mj], expHiN, [mk,ml])

#                             PI.cluster_H1j_vec[idx] = mel_H1j
#                             PI.cluster_HiN_vec[idx] = mel_HiN
#                             PI.cluster_expH1j_vec[idx] = mel_expH1j
#                             PI.cluster_expHiN_vec[idx] = mel_expHiN
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# function init_MCpath_PIMC(PI::PathIntegral, status::String)
#     P,N = PI.P, PI.N

#     # ordered: all rotors on the m=0 state
#     # does nothing

# 	# random:
#     if (status == "random")
#         # changing the middle beads (first and last bead do not change)        
# 		if (N == 2)
# 			for p = 2:P-2
# 				accept_proposal = false
# 				while accept_proposal==false                    
#                     mi,mj = PI.MCpath[p-1, 1:2]
#                     mk,ml = rand([1:PI.Nstates;], 2)
#                     m_el = cluster_search(PI.cluster_expHij_vec, [mi,mj,mk,ml])
#                     if m_el != 0
#                         accept_proposal = true
# 						PI.MCpath[p, 1:2] = [mk,ml]
# 					end                    
# 				end
# 			end
# 			# condition for bead P-1
#             accept_proposal = false
#             while accept_proposal == false
#                 mi,mj = PI.MCpath[P-2, 1:2]
#                 mk,ml = rand([1:PI.Nstates;], 2)
#                 mo,mp = PI.MCpath[P, 1:2]
#                 m_el = cluster_search(PI.cluster_expHij_vec, [mi,mj,mk,ml]) * cluster_search(PI.cluster_expHij_vec, [mk,ml,mo,mp])
#                 if m_el != 0
#                     accept_proposal = true
#                     PI.MCpath[P-1, 1:2] = [mk,ml]
#                 end
#             end

# 		# N>2 planar rotors
# 		else
#             for p = 2:P-2
#                 aux = p%2 
# 				for n = 1+(aux) : 2 : N-1
#                     accept_proposal = false
#                     while accept_proposal == false
#                         mi,mj = PI.MCpath[p-1, n:n+1]
#                         mk,ml = rand([1:PI.Nstates;], 2)
#                         if n==1
#                             m_el = cluster_search(PI.cluster_expH1j_vec, [mi,mj,mk,ml])
#                         elseif (n>1) && (n<N-1)
#                             m_el = cluster_search(PI.cluster_expHij_vec, [mi,mj,mk,ml])
#                         elseif n==N-1
#                             m_el = cluster_search(PI.cluster_expHiN_vec, [mi,mj,mk,ml])
#                         end

#                         if m_el != 0
#                             accept_proposal = true
#                             PI.MCpath[p, n:n+1] = [mk,ml]
#                         end
#                     end
# 				end
#                 if (aux==1)
#                     PI.MCpath[p, 1] = PI.MCpath[p-1, 1]
#                     if (N%2==0)
#                         PI.MCpath[p, N] = PI.MCpath[p-1, N]
#                     elseif (N%2==1)
#                         PI.MCpath[p+1, N] = PI.MCpath[p, N]
#                     end
#                 end
#             end

#             # condition for bead P-1
#             for n = 2:2:N-1
#                 accept_proposal = false
#                 while accept_proposal == false
#                     mi,mj = rand([1:PI.Nstates;], 2)
#                     mk,ml = PI.MCpath[P, n:n+1]
#                     if (n>1) && (n<N-1)
#                         m_el = cluster_search(PI.cluster_expHij_vec, [mi,mj,mk,ml])
#                     elseif n==N-1
#                         m_el = cluster_search(PI.cluster_expHiN_vec, [mi,mj,mk,ml])
#                     end

#                     if m_el != 0
#                         accept_proposal = true
#                         PI.MCpath[P-1, n:n+1] = [mi,mj]
#                     end
#                 end
#             end
# 		end
#     end
# end

# function cluster_search(cluster_vec::Vector{Float64}, indices::Vector{Int64})
#     Nstates = Int64(length(cluster_vec)^(1/4))
#     return cluster_vec[map_vector2index(Nstates, indices)]
# end

function cluster_search(PI::PathIntegral, indices::Vector{Int64})
    return PI.cluster_expVij_vec[map_vector2index(PI.Nstates, indices)]
end


function cluster_PIMC_probs(PI::PathIntegral, coord::Vector{Int64})
    grid = PI.MCpath
	P, N = PI.P, PI.N
	p, n = coord
    m0 = PI.m_max + 1
    cluster_probs = zeros(Float64, PI.Nstates^4)
    # Notation:
    # I: cluster interacting with the i-th particle

    # p+n needs to be even
    if ( (p+n)%2 == 1 )
        error("Grid coordinates need to be (p,n) such that p+n is even")
    else
        # top:
        if (p==1)
            # particles i and j are held fixed
            # No I and J clusters
            mi = m0
            mj = m0
            # TOP-LEFT:
            if (n==1)
                for mk = 1:PI.Nstates
                    for ml = 1:PI.Nstates
                        K = [
                            mk, grid[p+2,n+1],
                            grid[p+3,n], grid[p+3,n+1]
                        ]
                        L = [
                            ml, grid[p+1,n+2],
                            grid[p+2,n+1], grid[p+2,n+2]
                        ]
                        # Kinetic energy contribution due to mi: (1,1); mj: (1,2); and mk: (3,1)
                        k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)*exp(-PI.tau*(mk-1-PI.m_max)^2)
                        prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                        idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                        cluster_probs[idx] = prob
                    end
                end            
                return cluster_probs ./ sum(cluster_probs) # normailzed

            # TOP-CENTER:
            elseif (n>1) && (n<N-1)
                for mk = 1:PI.Nstates
                    for ml = 1:PI.Nstates
                        K = [
                            grid[p+1,n-1], mk,
                            grid[p+2,n-1], grid[p+2,n]
                        ]
                        L = [
                            ml, grid[p+1,n+2],
                            grid[p+2,n+1], grid[p+2,n+2]
                        ]
                        # Kinetic energy contribution due to mi: (1,n); and mj: (1,n+1)
                        k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                        prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                        idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                        cluster_probs[idx] = prob
                    end
                end            
                return cluster_probs ./ sum(cluster_probs) # normailzed

            # TOP-RIGHT:
            elseif (n==N-1)
                # In this case N-1 has to be odd
                for mk = 1:PI.Nstates
                    for ml = 1:PI.Nstates
                        K = [
                                grid[p+1,n-1], mk,
                                grid[p+2,n-1], grid[p+2,n]
                        ]
                        L = [
                            grid[p+2,n], ml,
                            grid[p+3,n], grid[p+3,n+1]                    
                        ]
                        # Kinetic energy contribution due to mi: (1,1); mj: (1,2); and ml: (p+2,N)
                        k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                        prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                        idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                        cluster_probs[idx] = prob
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed
            end        
        
        # 2nd-bead:
        elseif (p==2)
            if (n>1) && (n<N-1)
                # 2nd-BEAD-Center: Cluster interacts with 4 other clusters
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            for ml = 1:PI.Nstates
                                I = [
                                    grid[p-1,n-1] , grid[p-1,n],
                                    grid[p,n-1], mi
                                ]
                                J = [
                                    grid[p-1,n+1] , grid[p-1,n+2],
                                    mj, grid[p,n+2]
                                ]
                                K = [
                                    grid[p+1,n-1] , mk,
                                    grid[p+2,n-1], grid[p+2,n]
                                ]
                                L = [
                                    ml , grid[p+1,n+2],
                                    grid[p+2,n+1], grid[p+2,n+2]
                                ]
                                # Kinetic energy contribution due to mk: (3,n); and ml: (3,n+1)
                                k_prob = exp(-PI.tau*(mk-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                                prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                                idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                                cluster_probs[idx] = prob
                            end
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed

            elseif (n==N-1)
                # 2nd-BEAD-RIGHT: differet condition. Cluster interacts with 3 other clusters
                # In this case N-1 has to be even
                mj = m0
                for mi = 1:PI.Nstates
                    for mk = 1:PI.Nstates
                        for ml = 1:PI.Nstates
                            I = [
                                grid[p-1,n-1] , grid[p-1,n],
                                grid[p,n-1], mi
                            ]
                            # J: no interaction mj = m0
                            K = [
                                grid[p+1,n-1] , mk,
                                grid[p+2,n-1], grid[p+2,n]
                            ]
                            L = [
                                grid[p+2,n], ml, 
                                grid[p+3,n], grid[p+3,n+1]
                            ]
                            # Kinetic energy contribution due to mk: (3,N-1); and ml: (3,N)
                            k_prob = exp(-PI.tau*(mk-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                            prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                            idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                            cluster_probs[idx] = prob
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed           
            end

        # center
        elseif (p>2) && (p<P-2)
            # CENTER-LEFT:
            # Cluster interacts with 4 other clusters
            if (n==1)
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            for ml = 1:PI.Nstates
                                I = [
                                    grid[p-2,n], grid[p-2,n+1],
                                    mi, grid[p-1,n+1]
                                ]
                                J = [
                                    grid[p-1,n+1] , grid[p-1,n+2],
                                    mj, grid[p,n+2]
                                ]
                                K = [
                                    mk, grid[p+2,n+1],
                                    grid[p+3,n], grid[p+3,n+1]
                                ]
                                L = [
                                    ml , grid[p+1,n+2],
                                    grid[p+2,n+1], grid[p+2,n+2]
                                ]
                                # Kinetic energy contribution due to mi: (p,1); and mj: (p,2)
                                k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                                prob = (k_prob*
                                    cluster_search(PI, [mi,mj,mk,ml])*
                                    cluster_search(PI, I)*
                                    cluster_search(PI, J)*
                                    cluster_search(PI, K)*
                                    cluster_search(PI, L))
                                idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                                cluster_probs[idx] = prob
                            end
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed

            # CENTER-CENTER:
            # Cluster interacts with 4 other clusters
            elseif (n>1) && (n<N-1)
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            for ml = 1:PI.Nstates                                
                                I = [
                                    grid[p-1,n-1] , grid[p-1,n],
                                    grid[p,n-1], mi
                                ]
                                J = [
                                    grid[p-1,n+1] , grid[p-1,n+2],
                                    mj, grid[p,n+2]
                                ]
                                K = [
                                    grid[p+1,n-1] , mk,
                                    grid[p+2,n-1], grid[p+2,n]
                                ]
                                L = [
                                    ml , grid[p+1,n+2],
                                    grid[p+2,n+1], grid[p+2,n+2]
                                ]
                                # Kinetic contribution
                                if (p%2 == 1)
                                    # Kinetic energy contribution due to mi: (p odd,n); and mj: (p odd,n+1)
                                    k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                                else
                                    # Kinetic energy contribution due to mk: (p+1 odd,n); and ml: (p+1 odd,n+1)
                                    k_prob = exp(-PI.tau*(mk-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                                end
                                prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)*cluster_search(PI, K)*cluster_search(PI, L)*k_prob
                                idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                                cluster_probs[idx] = prob
                            end
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed

            # CENTER-RIGHT:
            # Cluster interacts with 4 other clusters
            elseif (n==N-1)
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            for ml = 1:PI.Nstates                                
                                I = [
                                    grid[p-1,n-1] , grid[p-1,n],
                                    grid[p,n-1], mi
                                ]
                                J = [
                                    grid[p-2,n] , grid[p-2,n+1],
                                    grid[p-1,n], mj
                                ]
                                K = [
                                    grid[p+1,n-1] , mk,
                                    grid[p+2,n-1], grid[p+2,n]
                                ]
                                L = [
                                    grid[p+2,n], ml, 
                                    grid[p+3,n], grid[p+3,n+1]
                                ]                                
                                # Kinetic contribution
                                if (n%2 == 1)
                                    # Kinetic energy contribution due to mi: (p odd,n); mj: (p odd,n+1); and ml: (p+2 odd,n+1)
                                    k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                                else
                                    # Kinetic energy contribution due to mj: (p-1 odd,n+1); mk: (p+1 odd,n); and ml: (p+1 odd,n+1)
                                    k_prob = exp(-PI.tau*(mj-1-PI.m_max)^2)*exp(-PI.tau*(mk-1-PI.m_max)^2)*exp(-PI.tau*(ml-1-PI.m_max)^2)
                                end
                                prob = (k_prob*
                                    cluster_search(PI, [mi,mj,mk,ml])*
                                    cluster_search(PI, I)*
                                    cluster_search(PI, J)*
                                    cluster_search(PI, K)*
                                    cluster_search(PI, L))
                                idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                                cluster_probs[idx] = prob
                            end
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed
            end           
        
        # (P-2)th-bead
        elseif (p==P-2)
            # P-2 is always odd
            if (n==1)
                # (P-2)th-BEAD-LEFT: differet condition. Cluster interacts with 3 other clusters
                mk = m0
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for ml = 1:PI.Nstates
                            I = [
                                grid[p-2,n] , grid[p-2,n+1],
                                mi, grid[p-1,n+1], 
                            ]
                            J = [
                                grid[p-1,n+1] , grid[p-1,n+2],
                                mj, grid[p,n+2]
                            ]
                            # K: no interaction
                            L = [
                                ml, grid[p+1,n+2], 
                                grid[p+2,n+1], grid[p+2,n+2]
                            ]                                
                            # Kinetic energy contribution due to mi: (p-2 odd,1); and mj: (p-2 odd,2)
                            k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                            prob = k_prob*cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)*cluster_search(PI, L)
                            idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                            cluster_probs[idx] = prob
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed
            elseif (n>1) && (n<N-1)
                # Like the CENTER-CENTER
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            for ml = 1:PI.Nstates                                
                                I = [
                                    grid[p-1,n-1] , grid[p-1,n],
                                    grid[p,n-1], mi
                                ]
                                J = [
                                    grid[p-1,n+1] , grid[p-1,n+2],
                                    mj, grid[p,n+2]
                                ]
                                K = [
                                    grid[p+1,n-1] , mk,
                                    grid[p+2,n-1], grid[p+2,n]
                                ]
                                L = [
                                    ml , grid[p+1,n+2],
                                    grid[p+2,n+1], grid[p+2,n+2]
                                ]
                                # Kinetic energy contribution due to mi: (p-2 odd,n); and mj: (p-2 odd,n+1)
                                k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                                prob = k_prob*cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)*cluster_search(PI, K)*cluster_search(PI, L)
                                idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                                cluster_probs[idx] = prob
                            end
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed

            elseif (n==N-1)
                # (P-2)-BEAD-RIGHT: differet condition. Cluster interacts with 3 other clusters
                # In this case N-1 has to be odd
                ml = m0
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        for mk = 1:PI.Nstates
                            I = [
                                grid[p-1,n-1] , grid[p-1,n],
                                grid[p,n-1], mi
                            ]
                            J = [
                                grid[p-2,n] , grid[p-2,n+1],
                                grid[p-1,n], mj
                            ]
                            K = [
                                grid[p+1,n-1] , mk,
                                grid[p+2,n-1], grid[p+2,n]
                            ]
                            # Kinetic energy contribution due to mi: (p-2 odd,n); and mj: (p-2 odd,n+1)
                            k_prob = exp(-PI.tau*(mi-1-PI.m_max)^2)*exp(-PI.tau*(mj-1-PI.m_max)^2)
                            prob = k_prob*cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)*cluster_search(PI, K)
                            idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                            cluster_probs[idx] = prob
                        end
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed
            end

        # bottom
        elseif (p==P-1)
            # particles k and l are held fixed
            # No K and L clusters
            mk = m0
            ml = m0
            # BOTTOM-CENTER:
            if (n>1) && (n<N-1)
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        I = [
                            grid[p-1,n-1], grid[p-1,n],
                            grid[p,n-1], mi
                        ]
                        J = [
                            grid[p-1,n+1], grid[p-1,n+2],
                            mj, grid[p,n+2]
                        ]
                        # no Kinetic energy contribution
                        prob = cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)
                        idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                        cluster_probs[idx] = prob
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed

            # BOTTOM-RIGHT:
            elseif (n==N-1)
                # In this case N-1 has to be even
                for mi = 1:PI.Nstates
                    for mj = 1:PI.Nstates
                        I = [
                            grid[p-1,n-1], grid[p-1,n],
                            grid[p,n-1], mi
                        ]
                        J = [
                            grid[p-2,n], grid[p-2,n+1],
                            grid[p-1,n], mj
                        ]
                        # Kinetic energy contribution due to mj: (p-1 odd,N)
                        k_prob = exp(-PI.tau*(mj-1-PI.m_max)^2)
                        prob = k_prob*cluster_search(PI, [mi,mj,mk,ml])*cluster_search(PI, I)*cluster_search(PI, J)
                        idx = map_vector2index(PI.Nstates, [mi,mj,mk,ml])
                        cluster_probs[idx] = prob
                    end
                end
                return cluster_probs ./ sum(cluster_probs) # normailzed
            end
        end
    end
end

function cluster_Gibbs_samp_update(PI::PathIntegral, coord::Vector{Int64})
	P, N = PI.P, PI.N
	p, n = coord
    norm_probs = cluster_PIMC_probs(PI, coord)

    # defining the new cluster
    new_cluster_index = sample(1:PI.Nstates^4,ProbabilityWeights(norm_probs))
    mi,mj,mk,ml = map_index2vector(PI.Nstates, 4, new_cluster_index)

    # updating the MC path
    PI.MCpath[p,n] = mi
    PI.MCpath[p,n+1] = mj
    PI.MCpath[p+1,n] = mk
    PI.MCpath[p+1,n+1] = ml

    # updating the "loose" specific cases (corners)
    if (p==1)
        if (n==1)
            PI.MCpath[p+2,n] = mk
        elseif (n==N-1)
            PI.MCpath[p+2,n+1] = ml
        end
    elseif (p>1) && (p<P-1)
        if (n==1)
            PI.MCpath[p-1,n] = mi
            PI.MCpath[p+2,n] = mk
        elseif (n==N-1)
            PI.MCpath[p-1,n+1] = mj
            PI.MCpath[p+2,n+1] = ml
        end
    elseif (p==P-1)
        if (n==1)
            PI.MCpath[p-1,n] = mi
        elseif (n==N-1)
            PI.MCpath[p-1,n+1] = mj
        end
    end
end


###############################################################################
function calculateError_byBinning(arr::Vector{Float64})
    # Finding the average and standard error using the binning method
    # This method requires 2^n data points, this truncates the data to fit this
    workingNdim = floor(Int64, log(length(arr)) / log(2.0))
    trunc = floor(Int64, length(arr) - 2^workingNdim)
    mean = Statistics.mean(arr[trunc+1:length(arr)])
    standardError = maxError_byBinning(mean, arr[trunc+1:length(arr)], workingNdim - 6)
    return mean, standardError
end
###############################################################################
function maxError_byBinning(mean, data, workingNdim)
    if workingNdim <= 1
        error("Not enough points MC steps were used for the binning method, please increase the number of MC steps")
    end
    errors = zeros(workingNdim)
    errors[1] = errorpropagation(mean, data)

    for i = 2:workingNdim
        ndim = floor(Int64, length(data) / 2)
        data1 = zeros(ndim)

        for j = 1:ndim
            data1[j] = 0.5 * (data[2*j-1] + data[2*j])
        end
        data = data1
        errors[i] = errorpropagation(mean, data)
    end
    return maximum(errors)
end
###############################################################################
function errorpropagation(mean, data)
    ndim = length(data)
    errors = Statistics.std(data, corrected=false) / sqrt(ndim)
    return errors
end
###############################################################################


###############################################################################
end