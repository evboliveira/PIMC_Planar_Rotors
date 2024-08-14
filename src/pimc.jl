module PIMC

include("clusterization.jl")
include("sys_operators.jl")

using .SysOperators
using .Clusterization
import LinearAlgebra as la
using StatsBase
using Statistics
export PathIntegral, runMC

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

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
    Nequilibrate::Int64 = 1
    Nstates::Int64 = 2*m_max+1
    beta::Float64 = 1.0/T
    tau::Float64= beta/L
	P::Int64 = L
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


    V_arr::Vector{Float64} = zeros(Float64, MC_steps)
    V_MC::Float64 = 0.0 
    V_stdError_MC::Float64 = 0.0

    E_arr::Vector{Float64} = zeros(Float64, MC_steps)
    E_MC::Float64 = 0.0
    E_stdError_MC::Float64 = 0.0

	K_arr::Vector{Float64} = zeros(Float64, MC_steps)
    K_MC::Float64 = 0.0
    K_stdError_MC::Float64 = 0.0

    M_total_arr::Vector{Float64} = zeros(Float64, MC_steps)
    M_total_MC::Float64 = 0.0
    M_total_stdError_MC::Float64 = 0.0

    histo_total::Vector{Float64} = zeros(Int64, Nstates)
    histo_initial::Vector{Float64} = zeros(Int64, Nstates)
	
    m_corr::Vector{Float64} = zeros(Float64, MC_steps)

	ergodic::Bool = true
end

function runMC(PI::PathIntegral)#, initialState = "random", sampling = "Gibbs")
    ################################################################################################
    #	Performs the monte carlo integration to simulate the system.
    #	
    #	Parameters
    #	----------
    #	initialState : string, optional
    #		Selects the distribution of the initial state of the system. The allowed options are 
	#       random or ordered. The default is random.
    #	sampling : string, optional
    #		Selects the sampling method. The allowed options are Gibbs or Metropolis. The default is
	#       Gibbs.
    ################################################################################################	
	if (PI.L < 1)
        error("A minimum of 2 beads must be used")
    end
    # Number of extra beads
	if (PI.N == 2)		
		PI.P = PI.L +1
    elseif (PI.N > 2)
        PI.P = 2*PI.L + 1        
    end
	P_mid = PI.P÷2 + 1

	ops = RotorOps(m_max = PI.m_max, g = PI.g)
	Vij = ops.Vij
    Kij = ops.Kij
	Hij = 2*Kij + Vij
    # expVij = (exp_matrix(-PI.tau * Vij))
    expHij = (exp_matrix(-PI.tau * Hij))
	# expVij = exp_matrix(-PI.tau*Vij)
    # expKij = exp_matrix(-0.5*PI.tau * Kij)

    # rho = expKij * expVij * expKij
    # eps = 1e-10
    # rho[ rho .<= eps ] .= 0

    cluster_probs = prob_table(PI.Nstates, expHij)
    twoB_probs = twoB_prob_table(PI.Nstates, expHij)
    two_body_sample = false
	#testing
	# idx_surr = map_vector2index(PI.Nstates, [6,6,6,6])
	# m = twoB_probs[:, idx_surr]
	# count =1
	# for i in m
	# 	println(count, "  ", i)
	# 	count+=1
	# end

	# initializing the MC Path
	path = init_MCpath_PIGS(PI, cluster_probs, "ordered")    
	# display_path(PI.m_max, path, true)

    # Ergoticity counters
    ergo_count = 0
    ergo_tolerance = PI.MC_steps ÷ 10
	counter = 0
	counter_E = 0.0
	# sampling = "random"
	sampling = "sequential"

	for mc = 1:PI.MC_steps
		# println("MC sweep  ", mc)

		if sampling == "random"
			n_random_steps = PI.N*PI.P
            for mc_i = 1:n_random_steps
                p = rand(2:PI.P-1)
                n = rand(1 : PI.N)
				if PI.N == 2 && two_body_sample == true
					new_pair = twoB_gibbs_samp(PI, p, path, twoB_probs)
					path[p, :] = new_pair
				else			
					new_m = gibbs_samp(PI,[p,n],path,cluster_probs)
					path[p,n] = new_m
					if (PI.N > 2) && (n == 1)
						if (p%2 == 0)
							path[p+1, n] = path[p, n]
						elseif (p%2 == 1)
							path[p-1, n] = path[p, n]
						end
					elseif (PI.N > 2) && (n == PI.N)
						if ((n+p)%2 == 0)
							path[p+1, n] = path[p, n]
						elseif ((n+p)%2 == 1)
							path[p-1, n] = path[p, n]
						end
					end	
				end
            end
		
		elseif sampling == "sequential"
			i=1
			for p = 2:PI.P-1 # for PIGS only
				if PI.N == 2 && two_body_sample == true
					new_pair = twoB_gibbs_samp(PI, p, path, twoB_probs)
					path[p, :] = new_pair
				else			
					for n = 1:PI.N 
						new_m = gibbs_samp(PI,[p,n],path,cluster_probs)
						path[p,n] = new_m
						if (PI.N > 2) && (n == 1)
							if (p%2 == 0)
								path[p+1, n] = path[p, n]
							elseif (p%2 == 1)
								path[p-1, n] = path[p, n]
							end	
							
						elseif (PI.N > 2) && (n == PI.N)
							if ((n+p)%2 == 0)
								path[p+1, n] = path[p, n]
							elseif ((n+p)%2 == 1)
								path[p-1, n] = path[p, n]
							end
						end	
					end
				end			
			end
		end
		
		
		# to print the last but one bead uncomment the next line
        # println(path[PI.P-1, 1:PI.N] .- (PI.m_max + 1))
		# Updating the ground state energy estimators for PIGS
		
		if (mc >= PI.Nequilibrate)

			# Kinetic Energy Estimator

			K = 0
			M = 0
            # println(path[P_mid,:])
			for i=1:PI.N
				K += (path[P_mid,i] - PI.m_max -1)^2
                M += (path[P_mid, i] - PI.m_max - 1)^2
			end
            PI.K_arr[mc] = K
            PI.M_total_arr[mc] = M


            if (PI.N == 2)			
				E0 = mel(path[PI.P-1, 1:2], expHij * Hij, path[PI.P, 1:2])
				Z0 = mel(path[PI.P-1, 1:2], expHij, path[PI.P, 1:2])

			elseif (PI.N == 3)
				E0 = (mel(path[PI.P-2, 1:2], expVij * Vij, path[PI.P-1, 1:2]) / mel(path[PI.P-1, 1:2], expVij, path[PI.P, 1:2])
                    +
                    mel(path[PI.P-2, 2:3], expVij * Vij, path[PI.P-1, 1:2]) / mel(path[PI.P-1, 2:3], expVij, path[PI.P, 2:3])
					)

			else
				E0 = 0

				num_total_states = PI.Nstates^PI.N
				for index = 1:num_total_states
					Malpha = map_index2vector(PI.Nstates, PI.N, index)
										
					rho = 1
					gamma = 0
					for i=1:PI.N-1
						remain_Malpha = append!(Malpha[1:i-1],Malpha[i+2:PI.N])
						remain_path = append!(path[PI.P,1:i-1],path[PI.P,i+2:PI.N])

						if remain_Malpha == remain_path
							if (i%2==0)
								rho *= mel(path[PI.P-1, i:i+1], expVij, Malpha[i:i+1])
							end
							gamma += mel(Malpha[i:i+1], Vij, path[PI.P, i:i+1]) 								                          	                            
						end
					end
					E0 += rho*gamma		
				end
				
				Z0 = 1
				for i=1:PI.N-1
					if (i%2==0)
						Z0 *= mel(path[PI.P-1, i:i+1], expVij, path[PI.P, i:i+1])
					end
				end
			end

			## testing with middle bead
			# if (PI.N == 2)			
			# 	e0 = mel(path[PI.L+1-1, 1:2], expVij * Hij, path[PI.L+1, 1:2])
			# 	z0 = mel(path[PI.L+1-1, 1:2], expVij, path[PI.L+1, 1:2])			
			# 	E0 = e0 / z0
			# else
			# 	E0 = 0
			# 	num_total_states = PI.Nstates^PI.N
			# 	for index = 1:num_total_states
			# 		Malpha = map_index2vector(PI.Nstates, PI.N, index)
										
			# 		rho = 1
			# 		gamma = 0
			# 		for i=1:PI.N-1
			# 			remain_Malpha = append!(Malpha[1:i-1],Malpha[i+2:PI.N])
			# 			remain_path = append!(path[PI.L+1,1:i-1],path[PI.L+1,i+2:PI.N])

			# 			if remain_Malpha == remain_path
			# 				if (i%2==0)
			# 					rho *= (mel(path[PI.L+1-1, i:i+1], expVij, Malpha[i:i+1])
			# 					/
			# 					mel(path[PI.L+1-1, i:i+1], expVij, path[PI.L+1, i:i+1])
			# 				)
			# 				end
			# 				gamma += mel(Malpha[i:i+1], Vij, path[PI.L+1, i:i+1]) 								                          	                            
			# 			end
			# 		end
			# 		E0 += rho*gamma				
			# 	end
			# end			

            # if (E0 > -1) && (E0 < 0)
			# 	counter_E+=E0
			# 	counter+=1

            # # elseif (E0 < -4)
            # #     println("E0=", E0, "   ", path[PI.P-1, 1:2])
			# end

			PI.E_arr[mc] = E0
		end


		# println(path[PI.P-1,:])
        # display_path(PI.m_max, path, true)
		# println("E0=", E0)
		# wait_for_key("press any key to continue")
	end
	
	# if (ergo_count < ergo_tolerance)
	# 	PI.ergodic = true
    # end
	# println(counter_E/counter)

    # meanE, stdErrE = calculateError_byBinning(PI.E_arr[PI.Nequilibrate+1:PI.MC_steps])
    # PI.E_MC = meanE
    # PI.E_stdError_MC = stdErrE

	meanK, stdErrK = calculateError_byBinning(PI.K_arr[PI.Nequilibrate+1:PI.MC_steps])
    PI.K_MC = meanK
    PI.K_stdError_MC = stdErrK

    meanM_total, stdErrM_total = calculateError_byBinning(PI.M_total_arr[PI.Nequilibrate+1:PI.MC_steps])
    PI.M_total_MC = meanM_total
    PI.M_total_stdError_MC = stdErrM_total

	# display_path(PI.m_max, path, true)
end

function prob_table(Nstates::Int64,rho_matrix::Array{Float64,2})
	probs = zeros(Float64, Nstates^4)
	for mi = 1:Nstates
		for mj = 1:Nstates
			for mip = 1:Nstates
				for mjp = 1:Nstates
					mel = rho_matrix[map_vector2index(Nstates,[mi,mj]),map_vector2index(Nstates,[mip,mjp])]
					index = map_vector2index(Nstates,[mi,mj,mip,mjp])
					probs[index] = mel
				end
			end
		end
	end
	return probs
end

function twoB_prob_table(Nstates::Int64, rho_matrix::Array{Float64,2})
    probs = zeros(Float64, (Nstates^2, Nstates^4))
    for mi_p1 = 1:Nstates
        for mj_p1 = 1:Nstates
            for mi_p2 = 1:Nstates
                for mj_p2 = 1:Nstates

					#Index of the surrounding pairs
					idx_surr = map_vector2index(Nstates, [mi_p1, mj_p1, mi_p2, mj_p2])
					norm = 0.0

                    for mi_pt = 1:Nstates
                        for mj_pt = 1:Nstates
                            #Index of the target pair
                            idx_t = map_vector2index(Nstates, [mi_pt, mj_pt])

                            mel1 = rho_matrix[map_vector2index(Nstates, [mi_p1, mj_p1]), map_vector2index(Nstates, [mi_pt, mj_pt])]
                            mel2 = rho_matrix[map_vector2index(Nstates, [mi_pt, mj_pt]), map_vector2index(Nstates, [mi_p2, mj_p2])]
                            probs[idx_t, idx_surr] = mel1 * mel2
                            norm += probs[idx_t, idx_surr]
                        end
                    end

					if norm != 0
                        probs[:, idx_surr] = probs[:, idx_surr] / norm
					end
                end
            end
        end
    end
	# for j=1:Nstates^4
	# 	println(probs[:,j])
	# end
	return probs
end


function init_MCpath_PIGS(PI::PathIntegral, probs::Vector{Float64}, status::String)
	P,N = PI.P, PI.N
    m0 = PI.m_max+1
    path = ones(Int64, (P, N)) * (m0-1)
	if (status == "ordered")
        return path
	else
		#middle beads
		path[2:P-1,1:N] .+= 1
        path[P-1, 1] = m0
		if (N%2==1)
			path[2,N] = m0
		else
            path[P-1, N] = m0
		end
		for p=2:P-1
			for n = 1:N 
				new_m = gibbs_uniform_samp(PI,[p,n],path,probs)
				path[p,n] = new_m
				if (N>2)
					if (n==1)
						if (p%2 == 0)
							path[p+1, n] = path[p, n]
						elseif (p%2 == 1)
							path[p-1, n] = path[p, n]
						end
					elseif (n==N)
						if (N>2) && ((n+p)%2 == 0)
							path[p+1, n] = path[p, n]
						elseif (N>2) && ((n+p)%2 == 1)
							path[p-1, n] = path[p, n]
						end
					end
				end
			end
        end
        # display(PI.m_max,path,false)
		return path
	end
end

function gibbs_samp(PI::PathIntegral,
	target_coords::Vector{Int64},
	MCpath::Array{Int64,2},
	cluster_probs::Vector{Float64})
	
	Nstates = PI.Nstates
	m_max = PI.m_max
	tau = PI.tau	

	p,n = target_coords
	P,N = PI.P, PI.N

	clusters = cluster_PIGS(target_coords,[P,N])
	if (typeof(clusters)==Nothing) || ((p==1) || (p==P))
		# no updates
		return MCpath[p,n]
	else
		update_probs = ones(Float64, Nstates)
        for m = 1:Nstates
			for braket_coords in clusters
				braket = [ MCpath[coord[1],coord[2]] for coord in braket_coords]
                braket[1] = m
				index_a = map_vector2index(Nstates, braket)
				mel_a = cluster_probs[index_a]

                # kinetic energy condition of the end particles
				k = 1
				aux = 1
                for coord in braket_coords
					if (coord[2] == 1 || coord[2] == N)
                        k *= exp(-0.5 * PI.tau * (braket[aux] - PI.m_max - 1)^2)
					end
					aux += 1
				end
				update_probs[m] *= mel_a * k
			end
        end
		
		### New condition to check probs
        if (sum(update_probs) == 0)
			println(update_probs)
			println(clusters)
			println(p,n)
			display_path(m_max,MCpath,false)
		end
        # norm_probs = la.normalize(update_probs) # WRONG!!!!!
		norm_probs = update_probs./sum(update_probs)
		# Updating the index states
		new_m = sample(1:Nstates,ProbabilityWeights(norm_probs))
		return new_m
	end	
end

function cluster_PIGS(coord::Vector{Int64}, grid_dim::Vector{Int64})
	# beads p=1 and p=P are kept unchanged, equal to the trial function |0>
    P,N = grid_dim
	p,n = coord

	## FOR N=2 PARTICLES
	if (N==2)
		if (p==1)
			return nothing
		elseif (p>1) && (p<P-1)
			if (n==1)
				cluster1 = [[p, n], [p, n + 1], [p - 1, n], [p - 1, n + 1]]
				cluster2 = [[p, n], [p, n + 1], [p + 1, n], [p + 1, n + 1]]
				return [cluster1, cluster2]
			elseif (n==2)
				cluster1 = [[p, n], [p, n - 1], [p - 1, n], [p - 1, n - 1]]
				cluster2 = [[p, n], [p, n - 1], [p + 1, n], [p + 1, n - 1]]
				return [cluster1, cluster2]
			end
		elseif (p==P-1)
			if (n==1)
				cluster1 = [[p, n], [p, n + 1], [p - 1, n], [p - 1, n + 1]]
				return [cluster1]
			elseif (n==2)
				cluster1 = [[p, n], [p, n - 1], [p - 1, n], [p - 1, n - 1]]
				return [cluster1]
			end
		elseif (p==P)
			return nothing
		end

	## FOR N>2 PARTICLES
	elseif (N>2)
		if (p==1)
			return nothing
		elseif (p>1) && (p<P-1)			
			if (n==1) 
				# C-type cluster
				if (p%2 == 0)
					# p even
					cluster1 = [[p,n], [p,n+1], [p-1,n], [p-1,n+1]]
					# [p,n] = [p+1,n]
					cluster2 = [[p,n], [p+1,n+1], [p+2,n], [p+2,n+1]]
					return [cluster1, cluster2]
				else
					# p odd
					cluster1 = [[p,n], [p,n+1], [p+1,n], [p+1,n+1]]
					# [p,n] = [p-1,n]
					cluster2 = [[p,n], [p-1,n+1], [p-2,n], [p-2,n+1]]
					return [cluster1, cluster2]
				end

			elseif (1<n<N) 
				# Z-type cluster
				if ((p+n)%2 == 0)					
					cluster1 = [[p,n], [p,n+1], [p+1,n], [p+1,n+1]]
					cluster2 = [[p,n], [p,n-1], [p-1,n], [p-1,n-1]]
					return [cluster1, cluster2]

				# S-type cluster
				elseif ((p+n)%2 == 1)					
					cluster1 = [[p,n], [p,n-1], [p+1,n], [p+1,n-1]]
					cluster2 = [[p,n], [p,n+1], [p-1,n], [p-1,n+1]]
					return [cluster1, cluster2]
				end
			
			elseif (n==N)
				# invC-type cluster
				if (N%2 == 1)
					if (p==2)
						return nothing
					elseif (p!=2)
						if (p%2==1)
							cluster1 = [[p,n], [p,n-1], [p-1,n], [p-1,n-1]]
							cluster2 = [[p,n], [p+1,n-1], [p+2,n], [p+2,n-1]]
							return [cluster1, cluster2]
						elseif (p%2==0)
							cluster1 = [[p,n], [p,n-1], [p+1,n], [p+1,n-1]]
							cluster2 = [[p,n], [p-1,n-1], [p-2,n], [p-2,n-1]]
							return [cluster1, cluster2]
						end
					end	

				elseif (N%2 == 0)
					if (p%2==0)
						cluster1 = [[p,n], [p,n-1], [p-1,n], [p-1,n-1]]
						cluster2 = [[p,n], [p+1,n-1], [p+2,n], [p+2,n-1]]							
						return [cluster1, cluster2]
					elseif (p%2==1)
						cluster1 = [[p,n], [p,n-1], [p+1,n], [p+1,n-1]]
						cluster2 = [[p,n], [p-1,n-1], [p-2,n], [p-2,n-1]]
						return [cluster1, cluster2]
					end				
				end
			end	
			
		elseif (p==P-1)			
			if (n==1) 
				return nothing

			elseif (1<n<N) 
				# Z-type cluster
				if (n%2 == 0)
					cluster1 = [[p,n], [p,n-1], [p-1,n], [p-1,n-1]]
					return [cluster1]

				# S-type cluster
				elseif (n%2 == 1)
					cluster1 = [[p,n], [p,n+1], [p-1,n], [p-1,n+1]]
					return [cluster1]
				end
			
			elseif (n==N)
				# invC-type cluster
				if (N%2 == 1)
					if (p==2)
						return nothing
					elseif (p!=2)
						cluster1 = [[p,n], [p-1,n-1], [p-2,n], [p-2,n-1]]
						return [cluster1]
					end	

				elseif (N%2 == 0)						
					return nothing		
				end
			end
			
		elseif (p==P)
			return nothing
		end
	end
end

function twoB_gibbs_samp(PI::PathIntegral,
    stage::Int64,
    MCpath::Array{Int64,2},
    twoB_probs::Array{Float64,2})

    Nstates = PI.Nstates
    m_max = PI.m_max
    tau = PI.tau

    p = stage
    idx_surr = map_vector2index(Nstates, [MCpath[p-1, 1], MCpath[p-1, 2], MCpath[p+1, 1], MCpath[p+1, 2]])
    
	# Updating the index states
	new_pair_idx = sample(1:Nstates^2, ProbabilityWeights(twoB_probs[:,idx_surr]))
    m1 = (new_pair_idx-1) ÷ Nstates + 1
    m2 = (new_pair_idx-1) % Nstates +1
	new_pair = [m1,m2]
	return new_pair
end

function gibbs_uniform_samp(PI::PathIntegral,
    target_coords::Vector{Int64},
    MCpath::Array{Int64,2},
    cluster_probs::Vector{Float64})

    Nstates = PI.Nstates
    m_max = PI.m_max
    tau = PI.tau

    p, n = target_coords
    P, N = PI.P, PI.N
    clusters = cluster_PIGS(target_coords, [P, N])
    if (typeof(clusters) == Nothing) || ((p == 1) || (p == P))
        # no updates
        return MCpath[p, n]
    else
        update_probs = ones(Float64, Nstates)
        for braket in clusters
            mi, mj, mip, mjp = [MCpath[coord[1], coord[2]] for coord in braket]
			for m = 1:Nstates
                mi = m
                index_a = map_vector2index(Nstates, [mi, mj, mip, mjp])
                mel_a = cluster_probs[index_a]
				if (mel_a != 0.)
                	update_probs[m] *= 1.
				else
					update_probs[m] *= 0.
				end
            end
        end

        ### New condition to check probs
        if (sum(update_probs) == 0)
			println(update_probs)
			println(clusters)
			println(p,n)
			display_path(m_max,MCpath,false)
		end
        # norm_probs = la.normalize(update_probs) # WRONG!!!!!
		norm_probs = update_probs./sum(update_probs)
        # Updating the index states
        new_m = sample(1:Nstates, ProbabilityWeights(norm_probs))
        return new_m
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