include("src/sys_operators.jl")
# include("src/pimc_cluster_samp.jl")
include("src/pimc.jl")
include("src/clusterization.jl")

using .Clusterization
using .PIMC
using .SysOperators
using LinearAlgebra
using Plots
using Printf
using Dates

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

MC_steps = 1000
N = 2
m_max = 5
g = 1.0
T = 0.1
L = 50
initial_state_type = "ordered"
sampling_type = "random"

# ####### Usual sampling##########################################
# PI = PathIntegral(N=N, m_max=m_max, g=g, L=L, MC_steps=MC_steps, T=T)

# runMC(PI)
# str = @sprintf("tau=%.4f",PI.tau)*@sprintf("      E0=%.4f",PI.E_MC)
# println(str)

# For different L's
size = 50
L_list = Int64.(round.(10 .^(range(start=log10(10),stop=log10(L),length=size))))
open("results/main.txt", "w") do file    
	for l in L_list
		PI = PathIntegral(N=N, m_max=m_max, g=g, L=l, MC_steps=MC_steps, T=T)
        runMC(PI)
        ans = string(PI.tau," ",PI.E_MC,"\n")
        write(file, ans)
		str = @sprintf("tau=%.4f",PI.tau)*@sprintf("      K=%.4f",PI.E_MC)
		println(str)
		# display(PI.E_arr[1:500:MC_steps])
		# wait_for_key("press any key to continue")
	end
end
linux_cmd = "gnuplot"
path = "plot.gnuplot"
run(`$linux_cmd $path`)

# Cluster sampling
# PI = PathIntegral(N=N, m_max=m_max, g=g, L=L, MC_steps=MC_steps, T=T)
# runMC_cluster(PI, initial_state_type, sampling_type)

# For different L's
# size = 20
# L_list = Int64.(round.(10 .^(range(start=log10(100),stop=log10(L),length=size))))
# open("results/main.txt", "w") do file    
# 	for l in L_list
# 		PI = PathIntegral(N=N, m_max=m_max, g=g, L=l, MC_steps=MC_steps, T=T)
# 		# PI.ergodic = false
# 		# while (PI.ergodic == false)
# 		# 	runMC_cluster(PI, initial_state_type, sampling_type)
# 		# end
# 		runMC_cluster(PI, initial_state_type, sampling_type)
#         ans = string(PI.tau," ",PI.E_MC,"\n")
#         write(file, ans)
# 		str = @sprintf("tau=%.4f",PI.tau)*@sprintf("      E0=%.4f",PI.E_MC)
# 		println(str)
# 		PI.display_path(m_max, PI.MCpath[PI.P-2:PI.P])
# 	end
# end
# linux_cmd = "gnuplot"
# path = "plot.gnuplot"
# run(`$linux_cmd $path`)



##### Testing the SysOperators
# PI = PathIntegral(N=N, m_max=m_max, g=g, L=L, MC_steps=MC_steps, T=T)
# ops = RotorOps(m_max = PI.m_max, g = PI.g)
# Vij = ops.Vij
# Kij = ops.Kij
# Hij = Vij + Kij
# expVij = exp_matrix(-PI.tau*Vij)
# expHij = exp_matrix(-PI.tau * Hij)
# # display(Vij)
# # display(expVij)
# # display(expHij)
# # for i=(1:(2*m_max+1)^2)
# #     for j = (1:(2*m_max+1)^2)
# # 		if (expVij[i,j] < 0)
# # 			println(
# # 			"negative"
# # 			)
# # 		end
# #     end
# # end
# est = expVij*Vij
# norm = est ./ expVij
# println(norm[tensor_index(PI.Nstates, [5, 7]), tensor_index(PI.Nstates, [6, 6])])
# ##### Testing the SysOperators

# For different L values
# Lmin = 49
# Lmax = 81
# nskip = 4
# L_list = reverse([l for l = Lmin : nskip : Lmax])

# info_string = (
# 	"MC_steps="*string(MC_steps)*"_"
# 	*"N="*string(N)*"_"
# 	*"m_max="*string(m_max)*"_"
# 	*"g="*string(g)*"_"
# 	*"T="*string(T)
# )

# out_data = open("results/tau_x_E0(N="*string(N)*").txt", "w")
# write(out_data,"#"*info_string)
# write(out_data, "#tau E_MC E_stdError_MC\n")

# for l in L_list
#     PI = PathIntegral(N=N, m_max=m_max, g=g, L=l, MC_steps=MC_steps,T=T)
#     runMC(PI)
# 	println(out_data, string(PI.tau)*" "*string(PI.E_MC)*" "*string(PI.E_stdError_MC))
# end
# close(out_data)
