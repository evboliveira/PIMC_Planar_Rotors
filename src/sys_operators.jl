module SysOperators
import LinearAlgebra as la
export RotorOps, exp_matrix, id_op

################################################################################
@kwdef struct RotorOps
####################################################################################################
#        Parameters
#        ----------
#        m_max : int
#            The maximum size of the free rotor eigenstate basis used to construct
#            the rotational density matrix.
#        g : float
#            Interaction strength between the rotors.
####################################################################################################
	m_max::Int64
	g::Float64
    Ngrid::Int64 = 2*m_max+1
    M2::Array{Int64,2} = Msquare_op(Ngrid)
    Ep::Array{Int64,2} = create_op(Ngrid)
    Em::Array{Int64,2} = annihilate_op(Ngrid)
    Kij::Array{Float64,2} = 0.5*la.kron(M2, id_op(Ngrid)) + 0.5*la.kron(id_op(Ngrid),M2)
    K1j::Array{Float64,2} = la.kron(M2, id_op(Ngrid)) + 0.5*la.kron(id_op(Ngrid),M2)
    KiN::Array{Float64,2} = 0.5*la.kron(M2, id_op(Ngrid)) + la.kron(id_op(Ngrid),M2)
	#dipole term
    Vij::Array{Float64,2} = -0.25*g*(3*la.kron(Ep,Ep)+la.kron(Ep,Em)+la.kron(Em,Ep)+3*la.kron(Em,Em))#.+0.
    #quadrupole term
	#Vij::Array{Float64,2} = 0.5 * g * (la.kron(Ep^2, Ep^2) + la.kron(Em^2, Em^2))#.+0.
end

function id_op(size::Int64)
    # "Creation" operator for a single rotor
    id = zeros(Int64, (size, size))
    for i = 1:size
        for j = 1:size
            if (i == j)
                id[i, j] = 1.0
            end
        end
    end
    return id
end

function Msquare_op(size)
	# Kinectic term operator for a single rotor
	M2_op = zeros(Int64,(size,size))
	for i = 1:size
		M2_op[i,i] = (i-1-(size-1)/2)^2
	end
	return M2_op
end

function create_op(size::Int64)
    # "Creation" operator for a single rotor
    E_p = zeros(Int64,(size,size))
    for i = 1:size
        for j = 1:size
            if (i == j + 1)
                E_p[i,j] = 1.0
            end
        end
    end
    #Normalization condition due to the truncation
    E_p[1,size]=1
    return E_p
end

function annihilate_op(size::Int64)
    # "Creation" operator for a single rotor
    E_m = zeros(Int64, (size, size))
    for i = 1:1:size
        for j = 1:1:size
            if (i == j - 1)
                E_m[i,j] = 1.0
            end
        end
    end
    #Normalization condition due to the truncation
    E_m[size,1]=1
    return E_m
end

function exp_matrix(matrix::Array{Float64,2})
	Msize = size(matrix,1)
	F = la.eigen(matrix)
	expM_diag = zeros(Float64,(Msize,Msize))
	for i = 1:Msize
		expM_diag[i,i] = exp(F.values[i])
	end
    # expM = F.vectors * (expM_diag * transpose(F.vectors))
    expM = round.(abs.(F.vectors * (expM_diag * transpose(F.vectors))), digits=18)
	return expM
    # return exp(matrix)
end

###############################################################################
end