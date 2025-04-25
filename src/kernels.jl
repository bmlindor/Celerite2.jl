	abstract type CeleriteKernel <: KernelFunctions.Kernel end 
	"""
	 The covariance function for data y with measurement error σ is 

		K(τ) = σ² × δ + ∑_j k_j(τ)

	where τ = |t_n - t_m| is the input, and δ is the Kronecker delta.
	A general Celerite kernel is : 
		k_j(τ) = [a_j × cos{-d_j × τ}]+ [b_j × sin{-d_j × τ}] × e^{-c_j × τ} 

	A real kernel (where b_j=0 and d_j=0) is :
		k_j(τ) = a_j × exp(-c_j × τ)
	""" 
## Types and structures
	struct ComplexKernel{T} <: CeleriteKernel
	# celerite kernel
		a::Vector{T}
		b::Vector{T}
		c::Vector{T}
		d::Vector{T}
	end

	function ComplexKernel(a::T,b::T,c::T,d::T) where T<:Float64
		return ComplexKernel([a],[b],[c],[d])
	end

	function _get_coefficients(k::ComplexKernel) 
		return (zeros(0), zeros(0), k.a, k.b, k.c, k.d)
	end

	struct RealKernel{T} <: CeleriteKernel
	# real celerite kernel
		a::Vector{T}
		c::Vector{T}
	end

	function RealKernel(a::T,c::T) where T<:Real
		return RealKernel([a],[c])
	end

	function _get_coefficients(k::RealKernel) 
		return (k.a, k.c, zeros(0), zeros(0), zeros(0), zeros(0))
	end

	struct SHOKernel{T} <: CeleriteKernel
	# Simple harmonic oscillator (SHO) kernel
		logS0::Vector{T} 	# ∝ power spectral density at ω0
		logQ::Vector{T} 	# oscillator quality factor
		logω0::Vector{T} 	# frequency of undamped oscillator
	end

	function SHOKernel(logS0::T,logQ::T,logω0::T) where T<:Float64
		return SHOKernel([logS0],[logQ],[logω0])
	end

	function SHOKernel(;ρ::Float64,τ::Float64,stdev::Float64)
		# alternative parameterization
		# ρ::Vector{T} # the undamped period of the oscillator
		# τ::Vector{T} # the damping timescale of the process
		# σ::Vector{T} # the standard deviation of the process
		ω0 = 2pi/ρ
		Q = τ * ω0 * 0.5
		S0 = stdev^2 / ω0*Q
		return SHOKernel(log(S0),log(Q),log(ω0))
	end

	function _get_coefficients(k::SHOKernel) 
		eps = 1e-5 # regularization parameter for numerical stability
		# overdamped if Q < 0.5
		S0 = exp(only(k.logS0)); Q = exp(only(k.logQ) ); ω0 =exp( only(k.logω0))
		# @assert Q > 0.0
		if Q < 0.5 
			f = sqrt(max(1 - 4 * Q^2,eps))
			a = 0.5 * S0 * ω0 * Q
			c = 0.5 * ω0 / Q
			return ([a * (1 + 1 / f), a * (1 - 1 / f)], [c * (1 - f), c * (1 + f)], 
				zeros(0), zeros(0), zeros(0),zeros(0))
		end
		f = sqrt(max(4 * Q^2 - 1,eps))
		a =  S0 * ω0 * Q
		c = 0.5 * ω0 / Q
		return (zeros(0), zeros(0), [a], [a / f], [c], [c * f])
	end

	struct RotationKernel{T} <: CeleriteKernel
		# mixture of two SHO kernels that models stellar rotation
		stdev::Vector{T} 	# standard deviation of process
		per::Vector{T} 		# primary period of variability
		Q0::Vector{T}		# quality factor of secondary oscillation
		dQ::Vector{T}		# difference in quality between the two modes 
		frac::Vector{T}		# secondary-to-primary-mode amplitude fraction 
		kernel::CeleriteKernel
	end

	function RotationKernel(stdev::T,per::T,Q0::T,dQ::T,frac::T) where T<:Real
		amp = stdev.^2 ./ (1 .+ frac)
		@assert 0.0 <= frac <= 1.0
		# First mode at period:
		Q1 = 0.5 .+ Q0 .+ dQ
		w1 = 4π .* Q1 ./ (per .* sqrt.(4 .* Q1.^2 .- 1))
		S1 = amp ./ (w1 .* Q1)

		# Second mode at half the period:
		Q2 = 0.5 .+ Q0 .+ dQ
		w2 = 8π .* Q2 ./ (per .* sqrt.(4 .* Q2.^2 .- 1))
		S2 = frac .* amp ./ (w2 .* Q2)

		kernel = SHOKernel(S1,Q1,w1) + SHOKernel(S2,Q2,w2)
		return RotationKernel([stdev],[per],[Q0],[dQ],[frac],kernel)
	end

	_get_coefficients(k::RotationKernel)=_get_coefficients(k.kernel)
# Kernel Operations
	struct CeleriteKernelSum{T} <: CeleriteKernel
	# sum of Celerite kernels
		kernels::T
	end

	function KernelFunctions.KernelSum(kernel1::CeleriteKernel, kernels::CeleriteKernel...)
		return CeleriteKernelSum((kernel1, kernels...))
	end

	function +(kernel1::CeleriteKernel ,kernels::CeleriteKernel...) 
		return CeleriteKernelSum((kernel1, kernels...))
	end

	function _get_coefficients(k::CeleriteKernelSum) 
	  ar = zeros(0);cr = zeros(0);
	  ac = zeros(0);bc = zeros(0);cc = zeros(0);dc = zeros(0);
	  for term in k.kernels
	      coeffs = _get_coefficients(term)
	      ar = vcat(ar, coeffs[1])
	      cr = vcat(cr, coeffs[2])
	      ac = vcat(ac, coeffs[3])
	      bc = vcat(bc, coeffs[4])
	      cc = vcat(cc, coeffs[5])
	      dc = vcat(dc, coeffs[6])
	  end
	  return ar, cr, ac, bc, cc, dc
	end

	struct CeleriteKernelProduct{T} <: CeleriteKernel
 	# product of two Celerite kernels
		kernels::T
	end

	function KernelFunctions.KernelProduct(kernel1::CeleriteKernel, kernel2::CeleriteKernel)
		return CeleriteKernelProduct((kernel1, kernel2))
	end

	function *(kernel1::CeleriteKernel ,kernel2::CeleriteKernel) 
		return CeleriteKernelProduct((kernel1, kernel2))
	end

	function _chain(x...)
		return (el for ind in x for el in ind)	
	end

	function _get_coefficients(k::CeleriteKernelProduct)
		ar1, cr1, ac1, bc1, cc1, dc1 = _get_coefficients(k.kernels[1])
		ar2, cr2, ac2, bc2, cc2, dc2 = _get_coefficients(k.kernels[2])

		nr1 = length(ar1) 
		nr2 = length(ar2)
		# Product of real terms:
		nr = nr1*nr2
		ar = zeros(nr) ; cr = zeros(nr)
		gen = product(zip(ar1,cr1),zip(ar2,cr2))
		for (i,((aj,cj),(ak,ck))) in enumerate(gen)
			ar[i] = aj * ak
			cr[i] = cj + ck
		end

		nc1 = length(ac1) 
		nc2 = length(ac2)
		# Product of real and complex terms:
		nc = nr1 * nc2 + nc1 * nr2 + 2 * nc1 * nc2
	  	ac = zeros(nc);bc = zeros(nc);cc = zeros(nc);dc = zeros(nc);
		gen = product(zip(ar1,cr1),zip(ac2,bc2,cc2,dc2))
		gen = _chain(gen, product(zip(ar2,cr2),zip(ac1,bc1,cc1,dc1)))
	  	for (i,((aj,cj),(ak,bk,ck,dk))) in enumerate(gen)
	  		ac[i] = aj * ak
	  		bc[i] = aj * bk
	  		cc[i] = cj + ck
	  		dc[i] = dk
	  	end
		# Product of complex and complex terms:
		gen = product(zip(ac1,bc1,cc1,dc1),zip(ac2,bc2,cc2,dc2))
		i0 = nr1 * nc2 + nc1 * nr2 + 1
	  	for (i,((aj,bj,cj,dj),(ak,bk,ck,dk))) in enumerate(gen)
		  	ac[i0 + 2*(i-1)] = 0.5 * (aj * ak + bj * bk)
	        bc[i0 + 2*(i-1)] = 0.5 * (bj * ak - aj * bk)
	        cc[i0 + 2*(i-1)] = cj + ck
	        dc[i0 + 2*(i-1)] = dj - dk

	        ac[i0 + 2*(i-1) + 1] = 0.5 * (aj * ak - bj * bk)
	        bc[i0 + 2*(i-1) + 1] = 0.5 * (bj * ak + aj * bk)
	        cc[i0 + 2*(i-1) + 1] = cj + ck
	        dc[i0 + 2*(i-1) + 1] = dj + dk
	    end
	    return ar,cr,ac,bc,cc,dc
	end

	struct CeleriteKernelDiff{T} <: CeleriteKernel
		kernel::T
	end

	function _get_coefficients(k::CeleriteKernelDiff)
		# for term in k.kernels
	    ar, cr, ac, bc, cc, dc = _get_coefficients(k)
        final_coeffs = ([-ar * cr ^2],
            [ac],
            [ac * (dc^2 - cc^2) + 2 * bc * cc * dc],
            [bc * (dc^2 - cc^2) - 2 * ac * cc * dc],
            [cc],
            [dc])
		return final_coeffs

	end
## Properties ##
	# Allow keyword arguments
	ComplexKernel(; a::Float64=0.0,b::Float64=0.0,c::Float64=0.0,d::Float64=0.0)=ComplexKernel(a,b,c,d)
	RealKernel(; a::Real=0.0,c::Real=0.0)=RealKernel(a,c)
	SHOKernel(;S0::Float64=0.0,Q::Float64=0.0,ω0::Float64=0.0)=SHOKernel(S0,Q,ω0) 
	SHOKernel(;S0::Float64=0.0,Q::Float64=0.0,w0::Float64=0.0)=SHOKernel(S0,Q,w0)
	SHOKernel(;ρ::Float64,τ::Float64,σ::Float64)=SHOKernel(;ρ=ρ,τ=τ,stdev=σ)
	RotationKernel(;stdev::Float64=1.5,per::Float64=3.45,Q0::Float64=1.3,dQ::Float64=1.05,frac::Float64=0.5) = RotationKernel(stdev,per,Q0,dQ,frac)	
	RotationKernel(;σ::Float64=1.5,per::Float64=3.45,Q0::Float64=1.3,dQ::Float64=1.05,frac::Float64=0.5) = RotationKernel(σ,per,Q0,dQ,frac)	

	# Overload size to length of components
	Base.size(k::ComplexKernel) = 4
	Base.size(k::SHOKernel) = 3
	Base.size(k::RealKernel) = 2
	Base.size(k::CeleriteKernelSum) = +(map(size,k.kernels)...)
	Base.size(k::CeleriteKernelProduct) = +(map(size,k.kernels)...)
	Base.size(k::RotationKernel) = 5

	get_kernel(k::ComplexKernel) = [only(k.a),only(k.b),only(k.c),only(k.d)]
	get_kernel(k::SHOKernel) = [only(k.logS0),only(k.logQ),only(k.logω0)]
	get_kernel(k::RealKernel) = [only(k.a),only(k.c)]
	get_kernel(k::CeleriteKernelSum) = cat(map(get_kernel,k.kernels)...,dims=1)
	get_kernel(k::CeleriteKernelProduct) = cat(map(get_kernel,k.kernels)...,dims=1)	
	get_kernel(k::RotationKernel) = [only(k.stdev),only(k.per),only(k.Q0),only(k.dQ),only(k.frac)]

	# Update kernel components
	function set_kernel!(kernel::ComplexKernel,vector)
		kernel.a .= [vector[1]] ; kernel.b .= [vector[2]]
		kernel.c .= [vector[3]] ; kernel.d .= [vector[4]]
	end

	function set_kernel!(kernel::RealKernel,vector)
		kernel.a .= [vector[1]] ; kernel.c .= [vector[2]]
	end

	function set_kernel!(kernel::SHOKernel,vector)
		kernel.logS0 .= [vector[1]] ; kernel.logQ .= [vector[2]] ; kernel.logω0 .= [vector[3]]
	end

	function set_kernel!(kernel::RotationKernel,vector)
		kernel.stdev .= [vector[1]] ; kernel.per .= [vector[2]] ; kernel.Q0 .= [vector[3]]
		kernel.dQ .= [vector[4]] ; kernel.frac .= [vector[5]] 
	end

 	function _set_kernels_in_operation!(k, vector::Vector)
 		ind = 1
 		for term in k.kernels
 			len = size(term)
 			set_kernel!(term,vector[ind:ind+len-1])
 			ind = ind + len
        end
 	end

 	set_kernel!(k::CeleriteKernelSum, vector::Vector)=_set_kernels_in_operation!(k,vector)
 	set_kernel!(k::CeleriteKernelProduct,vector::Vector) = _set_kernels_in_operation!(k,vector)

	# Aliases
	const HarmonicOscillatorKernel = SHOKernel
	const OrnsteinUhlenbeckKernel = RealKernel
	const DampedRandomWalkKernel = RealKernel
	const DRWKernel = RealKernel

## Kernel matrix ##
	function _get_value(k,x)
		# Compute values for the ∑_j k_j(τ) term of the covariance function. 
		ar, cr, ac, bc, cc, dc=_get_coefficients(k)
	    t = abs.(x)
	    k = zeros(size(x))
	    for i in 1:length(ar)
	        k .+= ar[i] .* exp.(-cr[i] .* t)
	    end
	    for i in 1:length(ac)
	        k .+= (ac[i] .* cos.(dc[i] .* t) .+ bc[i] .* sin.(dc[i] .* t)) .* exp.(-cc[i] .* t)
	    end
	    return k
	end

	(k::CeleriteKernel)(x) = _get_value(k,x)
	# function (k::CeleriteKernel)(x1,x2)
	# 	tau = x2 - x1
	# 	return _get_value(k,tau)
	# end

	function KernelFunctions.kernelmatrix(k::CeleriteKernel,x1::Vector,x2::Vector)
	    τ = broadcast(-, reshape(x1, length(x1), 1), reshape(x2, 1, length(x2)))
	    return _get_value(k,τ)
	end
	KernelFunctions.kernelmatrix(k::CeleriteKernel, x::AbstractVector) = kernelmatrix(k, x, x)
## Extra stuff
	Base.length(k::CeleriteKernelSum) = length(k.kernels)
	Base.length(k::CeleriteKernelProduct) = length(k.kernels)
	# Show components
	function Base.show(io::IO, k::RotationKernel)
		return print(
	    io, "Rotation Kernel (stdev = ", only(k.stdev), ", per = " , only(k.per),", Q0 = ", only(k.Q0), ", dQ = ", only(k.dQ), ", frac = ", only(k.frac),")")
	end
 
	function Base.show(io::IO, k::ComplexKernel)
		return print(
	    io, "Complex Celerite Kernel (a = ", only(k.a), ", b = ", only(k.b), ", c = ", only(k.c), ", d = ", only(k.d),")")
	end

	# function Base.show(io::IO, k::SHOKernel)
	# 	return print(
	#     io, "Simple Harmonic Oscillator Kernel (S0 = ", only(k.S0), ", Q = ", only(k.Q), ", ω0 = ", only(k.ω0),")")
	# end

	function Base.show(io::IO, k::RealKernel)
		return print(
	    io, "Real Celerite Kernel (a = ", only(k.a), ", c = ", only(k.c), ")")
	end

	function Base.show(io::IO,κ::CeleriteKernelSum)
		print(io, "Sum of $(length(κ)) Celerite kernels:")
		for k in κ.kernels 
			print(io,"\n", "\t", k)
		end
	end

	function Base.show(io::IO,κ::CeleriteKernelProduct)
		print(io, "Product of $(length(κ)) Celerite kernels:")
		for k in κ.kernels 
			print(io,"\n", "\t", k)
		end
	end	