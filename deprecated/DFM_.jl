function DFM_get_matrices(k::KernelFunctions.Kernel,x::Vector,yerr::Vector)
	ar, cr, ac, bc, cc, dc=Celerite2._get_coefficients(k)
	# Compute the dimensions of the problem:
	N=length(x)
	# Number of real and complex components:
	Jr = length(ar);    Jc = length(ac)
	# Rank of semi-separable components:
	J = Jr + 2*Jc
	# Sum over the diagonal kernel amplitudes and get diagonal components:
	A = yerr.^2 .+ sum(ar) .+ sum(ac) 
	arg = x * dc'
	cosdt = cos.(arg)
	sindt = sin.(arg)
	# With exponentials factored out, get variables from Eq. 43
	U = cat(ones(N) * ar' , 
	cosdt .* ac' + sindt .* bc' ,
	sindt .* ac' - cosdt .* bc' ,dims=2)
	V = cat(ones(N, Jr), cosdt, sindt, dims=2)
	# Compute time lag and ϕ:
	dx = x[2:N] - x[1:N-1]
	Phic = exp.(-dx * cc') 
	Phi = cat(exp.(-dx * cr'), Phic, Phic, dims=2)
	return A, U, V, Phi
end

function DFM_full_solve(k::KernelFunctions.Kernel,x::Vector,yerr::Vector=[0.0])
	@warn "full_solve_DFM : This is not indended for use with large datasets."
	A,U,V,Phi=get_matrices_DFM(k,x,yerr)
	N,J=size(U)
	# Compute the kernel matrix:
	K = zeros(N, N)
	# K::DenseArray{T, 2} = zeros(N, N)
	p = ones(J)
	@inbounds for m in 1:N
	vm = V[m, :]
	fill!(p, 1.0) 
	K[m, m] = A[m]
	@inbounds for n in m+1:N
	p = p .* Phi[n - 1, :]
	un = U[n, :]
	K[n, m] = sum(un .* vm .* p)
	K[m, n] = K[n, m]
	end
	end
	return K
end


function DFM_factor!(U::AbstractArray{T, 2},P::AbstractArray{T, 2},d::AbstractArray{T, 1},W::AbstractArray{T, 2}) where T

  N, J = size(U)
  Sn = zeros(J, J)

  W[1, :] ./= d[1]

  @inbounds for n in 2:N
    Sn += d[n - 1] .* W[n - 1, :] * transpose(W[n - 1, :])
    Sn = P[n - 1, :] .* Sn .* transpose(P[n - 1, :])
    tmp = transpose(U[n, :]) * Sn
    d[n] -= tmp * U[n, :]
    if (d[n] <= 0)
      throw("matrix not positive definite")
    end
    W[n, :] -= transpose(tmp)
    W[n, :] ./= d[n]
  end
  # return W
end

function DFM_solve!(U::AbstractArray{T, 2},P::AbstractArray{T, 2},d::AbstractArray{T, 1},W::AbstractArray{T, 2},Z::AbstractArray{T}) where T

  N, J = size(U)
  nrhs = size(Z, 2)
  Fn = zeros(J, nrhs)

  @inbounds for n in 2:N
    Fn = P[n - 1, :] .* (Fn + W[n - 1, :] * transpose(Z[n - 1, :]))
    Z[n, :] -= transpose(Fn) * U[n, :]
  end

  Z ./= d

  fill!(Fn, 0)
  @inbounds for n in N-1:-1:1
    Fn = P[n, :] .* (Fn + U[n + 1, :] * transpose(Z[n + 1, :]))
    Z[n, :] -= transpose(Fn) * W[n, :]
  end

end

function DFM_logpdf(k::KernelFunctions.Kernel,x::Vector,y::Vector,sigma::Vector)
  N = size(x, 1)
  y0 = copy(y)
  a, U, V, P = DFM_get_matrices(k, x, sigma.^2 .+ zeros(N))
  @time DFM_factor!(U, P, a, V)
  DFM_solve!(U, P, a, V, y0)
  return -0.5 * (transpose(y) * y0 + sum(log.(a)) + N * log(2*pi))
end