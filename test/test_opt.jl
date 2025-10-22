@testset "optimize" begin
	# filename = string("../..//Celerite2.jl/test/simulated_gp_data.txt")
	# data=readdlm(filename,comments=true)
	# x = data[:,1];
    # y = data[:,2];
    # yerr = data[:,3];
    # true_x = data[:,4];
    # true_y = data[:,5];
	# non periodic component
	Q = 1.0/sqrt(2.0) ; w0 = 3.0
	S0 = var(y) ./ (w0 * Q)
	comp_1=SHOKernel(log(S0),log(Q),log(w0))

	# periodic component
	Q = 1.0; w0 = 3.0
	S0 = var(y) ./ (w0 * Q)
	comp_2=SHOKernel(log(S0),log(Q),log(w0))

	kernel = comp_1 + comp_2
	gp=CeleriteGP(kernel,x,yerr)

	logL=logpdf(gp,y)

	# println("Difference in logLikelihood: ",orig_logL-logL)

	# Maximize the (log) marginal likelihood wrt. hyperparameters
	vector = get_kernel(kernel)
	mask = ones(Bool,size(kernel))
	mask[2] = false 	# We don't want to fit the first Q

	function nll(params)
		vector[mask] = params
		# build gp prior
		set_kernel!(gp.kernel,vector)
		# build finite gp
		gp_trial=CeleriteGP(gp.kernel,x,yerr)
        coeffs = _get_coefficients(gp.kernel)
		# if Celerite2._check_pos_def(coeffs) return -Inf
		# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
	end
	res_NM = optimize(nll, vector[mask],Optim.NelderMead())
	# println("minimized NM results:",res_NM.minimizer)
	res_LBFGS = optimize(nll, vector[mask],Optim.LBFGS())
	# println("minimized LBFGS results:",res_LBFGS.minimizer)
	vector[mask] = res_LBFGS.minimizer
	set_kernel!(gp.kernel,vector)
	@test maximum(abs.(res_LBFGS.minimizer - [ 3.1558461 , -2.05251577, -3.97153374,  2.20098169,  1.12798587])) <= 1e-4

	logL=logpdf(gp,y)
	@test isapprox(logL, 12.442495797758298)
	function nll_mean(params)
		vector_mean[mask_mean] = params
		# build gp prior
		set_kernel!(gp.kernel,vector_mean[2:end])
		# build finite gp
		gp_trial=CeleriteGP(gp.kernel,x,yerr,vector_mean[1])
		# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
	end
	vector_mean = [0.0;get_kernel(kernel)]
	mask_mean = ones(Bool,length(vector_mean))
	mask_mean[3] = false
	res_mean_LBFGS = optimize(nll_mean, vector_mean[mask_mean],Optim.LBFGS())

	# println(res_mean_LBFGS.minimizer)		# [3.1558052718461456,-2.052508255299794,-3.971541989475279,2.2009980296918314,1.127985820268802]
	# @test isapprox(round.(res_mean_LBFGS.minimizer,sigdigits=5),[ 3.1558461 , -2.05251577, -3.97153374,  2.20098169,  1.12798587])

    function full_math(gp::CeleriteGP,y_train::AbstractVector,x_train::AbstractVector,x::AbstractVector)
        μ = _k_matrix(gp,gp.x,x_train)' * inv(_k_matrix(gp,gp.x,gp.x)  + gp.Σy)  * (y_train )#.- mean(gp.x)) .+ mean(x)  
        C = _k_matrix(gp,x_train,x_train) - _k_matrix(gp,gp.x,x_train)' * (_k_matrix(gp,gp.x,gp.x)  + gp.Σy) *  _k_matrix(gp,gp.x,x_train)
        σ² =  diag(C)
        return μ,σ² 
    end

	@time "Reconstruct K w/ choleksy" myμ,myvar=mean_and_var(gp,y,true_x)
	@time "Full matrix inversion" μ_math, var_math = full_math(gp,y,x,true_x)
	α = apply_inverse(gp,y)
    @time "Ambikasaran method" μ_rec = predict(gp.kernel,x,y,true_x,α)
	@test maximum(abs.(μ_rec .- myμ)) <= 1e-5
	# @test maximum(abs.(μ_rec .- μ_math)) <= 1e-5
end
#=
# python equivalent
import numpy as np
import matplotlib.pyplot as plt
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import h5py 
data = h5py.File('research/Celerite2.jl/test/simulated_gp_data.h5','r')
y=np.array(data['data']['yobs'])
x=np.array(data['data']['xobs'])
yerr = np.array(data['data']['yerr'])
true_x = np.array(data['data']['true_x'])
true_y = np.array(data['data']['true_y'])
w0=3.0; Q=1.0/np.sqrt(2.0)
S0 = np.var(y) / (w0 * Q)
term1 = terms.SHOTerm(w0=w0, Q=Q,S0=S0)
w0=3.0; Q=1.0
S0 = np.var(y) / (w0 * Q)
term2 = terms.SHOTerm(w0=w0, Q=Q,S0=S0)
kernel= term1+term2
gp = celerite2.GaussianProcess(kernel, mean=0.0)
gp.compute(np.sort(x), yerr=yerr)
print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
Initial log likelihood: -148.45712640218147
def neg_log_like(params, gp):
    gp = set_params(params, gp)
    return -gp.log_likelihood(y)

def plot_prediction(gp):
    plt.plot(true_x, true_y, "k", lw=1.5, alpha=0.3, label="data")
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label="truth")
    μ, variance = gp.predict(y, t=true_x, return_var=True)
    sigma = np.sqrt(variance)
    plt.plot(true_x, μ, label="prediction")
    plt.fill_between(true_x, μ - sigma, μ + sigma, color="C0", alpha=0.2)

def set_params(params, gp):
	theta = np.exp(params[0:])
	gp.kernel =  terms.SHOTerm(S0=theta[0], Q=0.70710,w0=theta[1]) + terms.SHOTerm(S0=theta[2], Q=theta[3], w0=theta[4]) 
	gp.compute(np.sort(x), diag=yerr**2, quiet=True)
	return gp

initial_params = [ 0.0, 0.0, 0.0,0.0,0.0]
soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
opt_gp = set_params(soln.x, gp)
soln
soln.x
# array([ 3.1558461 , -2.05251577, -3.97153374,  2.20098169,  1.12798587])
print("Optimized log likelihood: {0}".format(opt_gp.log_likelihood(y)))
# 12.442495797758298
## similar to result above, not approx because of exponents/logarithms. 
=#
#=
# celerite.jl equivalent
Q = 1.0 / sqrt(2.0)
w0 = 3.0
S0 = var(y) ./ (w0 * Q)
kernel1 = celerite.SHOTerm(log(S0), log(Q), log(w0))
Q = 1.0
w0 = 3.0
S0 = var(y) ./ (w0 * Q)
orig_kernel = kernel1 + celerite.SHOTerm(log(S0), log(Q), log(w0))
orig_gp = celerite.Celerite(orig_kernel)
celerite.compute_ldlt!(orig_gp, x, yerr) 
	orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
orig_vector = celerite.get_parameter_vector(orig_gp.kernel)
mask = ones(Bool, length(orig_vector))
mask[2] = false  # Don't fit for the first Q
function nll_ldlt(params)
	orig_vector[mask] = params
	celerite.set_parameter_vector!(orig_gp.kernel, orig_vector)
	celerite.compute_ldlt!(orig_gp, x, yerr)
	return -celerite.log_likelihood_ldlt(orig_gp, y)
end

orig_res = Optim.optimize(nll_ldlt, orig_vector[mask], Optim.LBFGS())
orig_vector[mask] = Optim.minimizer(orig_res)
# println("minimized celerite LBFGS results:",exp.(orig_vector))

celerite.set_parameter_vector!(orig_gp.kernel, orig_vector)
orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)

@testset "fit without mean" begin
@test isapprox(orig_logL,logL)
# Sometimes the following lines will pass, other times will fail.
@test isapprox(orig_res.minimizer,res_LBFGS.minimizer)
@test maximum(abs.(gp.D- orig_gp.D)) < 1e-5 
@test maximum(abs.(gp.W- orig_gp.W)) < 1e-5 
@test maximum(abs.(gp.U- orig_gp.up)) < 1e-5
# @test isapprox(gp.U,orig_gp.up)
# @test isapprox(gp.W,orig_gp.W)
# @test isapprox(gp.D, orig_gp.D)
end

t1 = time()
μ_ldlt, variance_ldlt = celerite.predict_full_ldlt(orig_gp, y, true_x, return_var=true)
telapsed_ldlt = time() -t1
=#