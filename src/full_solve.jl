include("../src/cel2.jl")
using .Celerite2,Test,KernelFunctions,LinearAlgebra, Profile,Distributions,Optim
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
	Celerite2.zero_out!(gp.phi);	
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

# @time Celerite2._factorize!(gp2.coeffs, x, gp2.A, gp2.U,gp2.W, gp2.phi)
# z=
# # W0 = gp.W
# U0 = gp.U
# @time W_factor=Celerite2._factorize!(U0,V0,A0,phi0)
# @show W_factor
# ll_dfm=DFM_logpdf(SHO,x,y,yerr)
# Celerite2._cholesky_rewrite!(U0,V0,phi0,A0) segmentation fault. at some point V elements are Inf followed by NaNs


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
# 	U0,V0,phi0,A0=Celerite2._init_matrices(SHO,x,yerr)
# 	logdetK=Celerite2._factor_rewrite!(A0, U0,V0, phi0)
# 	y0 = copy(y)
# 	invKy =  Celerite2._solve!(A0,U0,V0,phi0,y0)
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
# @time  gp.D,gp.W,gp.up,gp.phi = celerite.cholesky!(coeffs..., x, yvar, gp.W, gp.phi, gp.up, gp.D)
# ll_agol2 = celerite.log_likelihood(gp, y)