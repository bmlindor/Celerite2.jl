# include("../src/cel2.jl")
# using .Celerite2,Test,KernelFunctions,LinearAlgebra, Profile,Distributions,Optim
# include("../../src/celerite.jl")
# include("../src/DFM_.jl")

# import .Celerite2: CeleriteGP,SHOKernel
using Random,Statistics,PyPlot
# Random.seed!(42)
rng=MersenneTwister(42)
# The input coordinates must be sorted
x = sort(cat(1, 3.8 .* rand(57), 5.5 .+ 4.5 .* rand(68);dims=1));
yerr = 0.08 .+ (0.22-0.08) .*rand(length(x));
y = 0.2.*(x.-5.0) .+ sin.(3.0.*x .+ 0.1.*(x.-5.0).^2) .+ yerr .* randn(length(x));

true_x=range(0,stop=10,length=1000)
true_y = 0.2 .*(true_x.-5) .+ sin.(3 .*true_x .+ 0.1.*(true_x.-5).^2);

# ax=subplot(111)
# ax.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
# ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
# ax.set_xlabel("x");ax.set_ylabel("y")
# ax.set_xlim(0, 10);ax.set_ylim(-2.5, 2.5);

# non periodic component
Q = 1.0/sqrt(2.0) ; w0 = 3.0
S0 = var(y) ./ (w0 * Q)
comp_1=Celerite2.SHOKernel(S0,Q,w0)

# periodic component
Q = 1.0; w0 = 3.0
S0 = var(y) ./ (w0 * Q)
comp_2=Celerite2.SHOKernel(S0,Q,w0)

kernel = comp_1 + comp_2
gp=Celerite2.CeleriteGP(kernel,x,yerr)

logL=logpdf(gp,y)
println("Initial logLikelihood: ",logL)


# coeffs = Celerite2._get_coefficients(kernel)
# println("coeffs:",coeffs)
## optimize GP
init_vector = Celerite2.get_kernel(gp.kernel)
mask = ones(Bool,size(kernel))
# We don't want to fit the first Q
mask[2] = false
function logL_wrapper1(params)
	init_vector[mask] = params
	logL_trial = 0.0
	if init_vector[5] < 0.0 
		logL_trial = -Inf
	else
	Celerite2.set_kernel!(kernel,init_vector)
	@show kernel
	Celerite2.zero_out!(gp.D)
	gp_trial=Celerite2.CeleriteGP(kernel,x,yerr)
	logL_trial=logpdf(gp_trial,y)
	end
	return -logL_trial
end

function logL_wrapper2(params)
	init_vector[mask] = params
	Celerite2.set_kernel!(gp.kernel,init_vector)
	Celerite2.zero_out!(gp.D);	
	Celerite2.zero_out!(gp.W);
	Celerite2.zero_out!(gp.ϕ);	
	Celerite2.zero_out!(gp.U);
	logL_trial=logpdf(gp,y)
	return -logL_trial
end

function logL_wrapper3(params)
	init_vector[mask] = params
	Celerite2.set_kernel!(kernel,init_vector)
	# Celerite2.zero_out!(gp.D)
	gp_trial=Celerite2.CeleriteGP(kernel,x,yerr)
	logL_trial=logpdf(gp_trial,y)
	return -logL_trial
end

x0 = init_vector[mask]
@time res_NM = optimize(logL_wrapper1, x0)
@show res_NM.minimizer

@time res_LBFGS = optimize(logL_wrapper1, x0,Optim.LBFGS())
@show res_LBFGS.minimizer

@time res_autodiff = optimize(logL_wrapper2, x0,LBFGS();autodiff = :forward)

# res3 = optimize(logL_wrapper3, init_vector[mask],Optim.BFGS())
# @show res3.minimizer



# res5 = optimize(logL_wrapper2, init_vector[mask],Optim.LBFGS())
# @show res5.minimizer

# res2 = optimize(logL_wrapper2, init_vector[mask])
# @show res2.minimizer


# N,x,yerr,y=Celerite2.make_test_data(30)
# # compare samples
# Nsample=5
# SHO=Celerite2.SHOKernel(exp(0.1), exp(2.0), exp(-0.5))
# gp2=Celerite2.CeleriteGP(SHO,x,yerr)
# logL=logpdf(gp2,y)
# noise = randn(rng,N)
# Celerite2._sample_data(gp,noise)
# sampled_yerr = rand(rng,gp,N,noise)

# Q = rand(Uniform(1,10),Nsample);
# # w0 = rand(Uniform(0.3,3),Nsample);
# w0= [1.0]
#  #rand(Uniform(-10,10),Nsample);
# figure()
# sampled_noise = zeros(Nsample,Nsample,N);
# for i in 1:length(Q)
# 	for j in 1:length(w0)
# 		S0 =0.01/w0[j]
# 	SHO=Celerite2.SHOKernel(S0, Q[i], w0[j])
# 	gp2=Celerite2.CeleriteGP(SHO,x,yerr)
# 	logL=logpdf(gp2,y)
# 	sampled_yerr = rand(rng,gp2,N)
# 	# push!(lnLs,logL)
# 	sampled_noise[i,j,:] .= sampled_yerr
# 	# diff = yerr.-sampled_noise[i,j,:]
# 	# if maximum(abs.(diff)) <= 2.5
# 	plot(w0[j]*x,sampled_noise[i,j,:],label=string(" Q = ",round(Q[i],sigdigits=2)," w0 =",round(w0[j],sigdigits=2)),ls="--",lw=1.5)
# # end
# end
# end
# legend()

# ax1=subplot(131);ax2=subplot(132);ax3=subplot(133)

# ylabel("sampled_noise",fontsize="x-large")
# xlabel("w0 * x",fontsize="x-large")
# title("Sampling from log-Uniform Q, and w0 values; S0=1e-4.",fontsize="x-large")
# legend()
# ax2.plot(exp(lnQ),yerr.-sampled_noise[i,:],label="Q")
# ax3.plot(exp(lnw0),yerr.-sampled_noise[i,:],label="w0")
# end

# @time Celerite2._factorize!(gp2.coeffs, x, gp2.A, gp2.U,gp2.W, gp2.ϕ)
# z=
# # W0 = gp.W
# U0 = gp.U
# @time W_factor=Celerite2._factorize!(U0,V0,A0,ϕ0)
# @show W_factor
# ll_dfm=DFM_logpdf(SHO,x,y,yerr)
# Celerite2._cholesky_rewrite!(U0,V0,ϕ0,A0) segmentation fault. at some point V elements are Inf followed by NaNs


# k2=celerite.SHOTerm(0.1, 2.0, -0.5)
# yvar = yerr.^2 + zeros(Float64, length(x))
# gp_old=celerite.Celerite(k2)
# celerite.compute_ldlt!(gp_old, x, yerr)
# # ll_agol = celerite.log_likelihood_ldlt(gp, y)

# y0 = celerite.simulate_gp_ldlt(gp_old,noise)

# figure()
# # plot(-0.5.*x, y0 - sampled_yerr)
# plot(y0,label="celerite")
# plot(sampled_yerr,label="mytest")
# # logL - ll_agol

# function try2()
# 	" This takes ~ 0.525543 seconds with 99.83% compilation time."
# 	U0,V0,ϕ0,A0=Celerite2._init_matrices(SHO,x,yerr)
# 	logdetK=Celerite2._factor_rewrite!(A0, U0,V0, ϕ0)
# 	y0 = copy(y)
# 	invKy =  Celerite2._solve!(A0,U0,V0,ϕ0,y0)
# 	logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))	
# 	return logL
# end
# sampled_yerr - y0
# time logL1 = try1()
# @time logL2 = try2()
# - comparing Agol and DFM solve functions
#     Agol full_solve is faster than DFM for N=126 example by 0.15 seconds on first run. 
#     ("Agol Semiseparable error: ", 1.5543122344752192e-15)
#     ("DFM Semiseparable error: ", 0.7288193185803731)
# celerite.compute!(gp, x, yerr)
# @time  gp.D,gp.W,gp.up,gp.ϕ = celerite.cholesky!(coeffs..., x, yvar, gp.W, gp.ϕ, gp.up, gp.D)
# ll_agol2 = celerite.log_likelihood(gp, y)


    # function mymean(post_gp,x::AbstractVector,y) 
    #     gp_prior = post_gp
    #     # mu0 = AbstractGPs.mean_vector(gp_prior.mean,x)
    #     mu =  Celerite2._k_matrix(gp_prior,x,gp_prior.x) * inv(Celerite2._k_matrix(gp_prior,gp_prior.x)  + gp_prior.Σy)  * (y .- mean(gp_prior.x))
    #     return mu
	#  end
    # function mycov(post_gp,x)
    #     gp_prior = post_gp
    #     C = Celerite2._k_matrix(gp_prior,x,x) - Celerite2._k_matrix(gp_prior,gp_prior.x,x)' * (Celerite2._k_matrix(gp_prior,gp_prior.x,gp_prior.x)  + gp_prior.Σy) *  Celerite2._k_matrix(gp_prior,gp_prior.x,x)
    # return C
    # end
    function mymean1(gp,y,x)
		# correct value but dont want to do inverse of K cuz it's expensive
        mu = diag(kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,x)) + gp.Σy) .* (y )#.- mean(gp.x))
        return mu
    end

	function full_Math_mu(gp,y,x)
		# correct value but dont want to do inverse of K cuz it's expensive
        mu =  diag(kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,x)) + gp.Σy) .* (y )#.- mean(gp.x))
        return mu
        end
  function full_Math_var(gp,x)
        B = kernelmatrix(gp.kernel,x,x)  - kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,gp.x)) * kernelmatrix(gp.kernel,x,gp.x)'
        return B
    end

    full_var = var(diag(full_Math_var(gp,collect(true_x))))
    function full_Math_cov(gp,x)
        C = kernelmatrix(gp.kernel,x,x)  - kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,gp.x) + gp.Σy) * kernelmatrix(gp.kernel,gp.x,x)
        return C
    end
    t2 = time()
    full_mu = full_Math_mu(gp,y,true_x)
    full_C = full_Math_cov(gp,true_x)
    full_var = diag(full_Math_cov(gp,true_x) )#.+ gp.Σy)
    telapsed_full=time() - t2

    t3 = time()
    mymu = mymean1(gp,y,true_x)
    # myC = mycov(gp,x)
    telapsed_equ=time() - t3
    # @testset "posterior" begin
    # 	@test isapprox(mymu,full_mu)
    # 	@test isapprox(myC,full_C)
    # end
    function test_posterior(gp,x_train,y_train,x)
    	K = kernelmatrix(gp.kernel,x_train,x_train)
    	K_s = kernelmatrix(gp.kernel,x_train,x)
    	Kss = kernelmatrix(gp.kernel,x,x)
    	K_inv = inv(K)
    	mus = K_s' * K_inv * y_train
    	covs = Kss - K' * K_inv * K_s
    	return mus, diag(covs + gp.Σy) 
    end
    
#=
function matmul(x,diag,y)
    if size(x,1) != size(y,1)
        throw("Dimension mismatch.")
    end
end
function _mat_mult_lower!(A::Vector{Float64},U::Array{Float64, 2},V::Array{Float64, 2},ϕ::Array{Float64,2},z::AbstractVector)
    J,N = size(U)
    f = zeros(Float64,J)
    for n =2:N
      f .= ϕ[:,n-1] .* (f .+ V[:,n-1] .* z[n-1])
      y[n] += dot(U[:,n-1],f)
    end
    return y
end

function _mat_mult_upper!(A::Vector{Float64},U::Array{Float64, 2},V::Array{Float64, 2},ϕ::Array{Float64,2},z::AbstractVector,y::AbstractVector)
    f = zeros(Float64,J)
    for n = N-1:-1:1
      f .= ϕ[:,n] .* (f .+  U[:,n] .* z[n+1])
      y[n] += dot(V[:,n],f)
    end
    return y
end
function _do_mat_mult!(A,U,V,ϕ,z)
    y = A .* z
    z = _mat_mult_upper!(U,V,ϕ,z,y)
    y = _mat_mult_lower(A,U,V,ϕ,z,y)   
end
        J,N = size(gp.U)
      z = zeros(Float64,N)
    z[1] = y[1]
      f = zeros(Float64,J)
      for n =2:N
        f .= gp.ϕ[:,n-1] .* (f .+ gp.W[:,n-1] .* z[n-1])
        z[n] = (y[n] - dot(gp.U[:,n], f))
      end
    # The following solves L^T.z = y for z:
      y = copy(z)
      fill!(z, zero(Float64))
      z[N] = y[N] / gp.D[N]
      fill!(f, zero(Float64))
      for n=N-1:-1:1
        f .= gp.ϕ[:,n] .* (f .+  gp.U[:,n+1] .* z[n+1])
        z[n] = y[n]/ gp.D[n] - dot(gp.W[:,n], f)
      end
      return z
function _solve_lower(U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},y::Vector{Float64})
    # Solve lower inverse of L.z = y for z:
    J,N = size(U)
    z = zeros(Float64,N)
    z[1] = y[1]
    f=zeros(Float64,J)
    @inbounds for n in 2:N
        f .= ϕ[:,n-1] .* (f .+ W[:,n-1] .* z[n-1])
        z[n] = (y[n] - dot(U[:,n],f))
    end
    return z
end

function _solve_upper!(U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},z::Vector{Float64})
    # Solve upper inverse of L' .z = y for z:
    J,N = size(U)
    f=zeros(Float64,J)
    @inbounds for n = N-1:-1:1
        f .= ϕ[:,n] .* (f .+ U[:,n+1] .* z[n+1])
        z[n] -=  dot(W[:,n],f)
    end
end
function _do_solve!(U,W,ϕ,y)
    z = _solve_lower(U,W,ϕ,y)   
    z ./= D 
    z = _solve_upper!(U,W,ϕ,z)
end
=#

#=
function _factor_rewrite!(D::Vector{Float64}, U,W,ϕ)
    # BL:  Rewrite of cholesky method, assuming that _init_matrices is called first.
    # @warn "Unstable?"
    J,N=size(U)
    W[:,1] ./= D[1] 

    # Allocate array for recursive computation of low-rank matrices:   
    S = zeros(J, J);
    Sk=0.0 ;  ϕj = 0.0 ; Dn = 0.0 ; Uk=0.0 ;  Uj = 0.0 ; Wj=0.0; tmp=0.0
    @inbounds for n in 2:N # what is diff b/w this loop form and for n in 2:N
        # Update S
        for j in 1:J
            ϕj=ϕ[j,n-1]
            Wj=W[j,n-1]
            for k in 1:j
                S[k,j] +=( D[n-1] .* Wj * W[k,n-1])
                S[k,j] = ϕj * ϕ[k,n-1] * S[k,j]
                # S[j,k] =  ϕj * ϕ[n-1,k] * (S[j,k] .+ (D[n-1] * Wj * W[n-1,k]))
            end 
        end
        # Update W and D
        # zero_out!(Dn)
        Dn = 0.0
        for j in 1:J
            Uj = U[j,n] 
            Wj = W[j,n]
            for k = 1:j-1
                Sk = S[k,j] ; 
                Uk = U[k,n]
                tmp = Uj * Sk
                Dn +=  Uk * tmp 
                Wj -= Uk * Sk 
                W[k,n] -= tmp
            end
            tmp = Uj * S[j,j]
            D[n] +=  Uj * tmp
            W[j,n] = Wj - tmp
        end
       Dn = D[n] 
       D[n] = Dn
       for j in 1:J
            W[j,n] /= Dn
       end
    end

    logdetK = 0.0
    for n in 1:N
    logdetK += log(D[n])
    end
    return logdetK
end

=#
