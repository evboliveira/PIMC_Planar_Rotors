module Clusterization

export map_vector2index, map_index2vector, cluster_PIMC, display_path, exp_value, mel

###############################################################################

function map_vector2index(dim::Int64, indices::Vector{Int64})
    # converts a number of any basis, with each algarism being a element of an 1-d array, to decimal
	# basis for indexation.
	indexation = 0
	for idx in indices
		if (idx >= 0) && (idx <= dim)
            # if the index count starts at i = 1 to i = dim, uncomment the following line
            indexation = indexation * dim + (idx-1)		
		else
			#in case of error
			error("Index higher than the number of dimensions!")
		end
	end
    # if the indexation count starts at 1, uncomment the following line
	indexation += 1
	return indexation
end

function map_index2vector(dim::Int64, num_entities::Int64, index::Int64)
    out = zeros(Int64, num_entities)
    aux = index - 1
    for n = 1:num_entities
        out[n] = aux ÷ dim^(num_entities - n) + 1
        aux = aux % dim^(num_entities - n)
    end
    return out
end


function display_path(m_max::Int64, path::Array{Int64,2}, display_ops::Bool = true)
    P, N = size(path)
    grid_m = path .- (m_max + 1)
	len = length(string(m_max)) + 1 #1 to account for the sign
	p_size = 4
    println(repeat("_", 3*len + (len+2) * N))
    line = "PATH"*repeat(" ", len*2-2)
	for n=1:N
		space = repeat(" ",len-length(string(n))+1)
		line*=space*string(n)*" "
	end
	println(line,"\n")
    # println(repeat("  ", len),repeat("-", length(line) - len-2))
	for p=1:P
		line = repeat(" ",p_size-length(string(p)))*string(p)*repeat(" ",len)
		op_line = repeat(" ",4*len)
		for n=1:N
			m = grid_m[p,n]
			el=""
			if (m==0)
                el *= repeat(" ", len)*"0"
			elseif (m>0)
                el *= repeat(" ", len - length(string(n))) * "+"*string(m)
			else
                el *= repeat(" ", len - length(string(n))) * string(m)
			end
			line*=el*" "
			if (N==2)
                op_line *= " " * repeat("=", len + 1)
			else
				if ((p+n)%2==0)
					op_line*=" "*repeat("=",len+1)
				else
					op_line *= repeat(" ", 2 * len)
				end
			end
		end
		println(line)
		if (p!=P) && (display_ops == true)
			println(chop(op_line, tail = len+1))
		end
	end
    println(repeat("_", 3 * len + (len + 2) * N), "\n")
end

function mel(bra::Vector{Int64}, two_part_Op::Array{Float64,2}, ket::Vector{Int64})
    Nstates = Int64(sqrt(size(two_part_Op)[1]))
	ind1 = map_vector2index(Nstates, [bra[1], bra[2]])
    ind2 = map_vector2index(Nstates, [ket[1], ket[2]])
	return two_part_Op[ind1,ind2]
end

# function cluster_PIMC(grid::Array{Float64,2}, target::Vector{Int64})
# 	P, N = size(grid)
# 	p, n = target

# 	# center-center
# 	I = []
# end

###############################################################################
end