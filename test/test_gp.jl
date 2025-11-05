@testset "gp" begin
	kernel=SHOKernel(log(0.1),log(2.0), log(0.5))
	gp=CeleriteGP(kernel,x,yerr)

	coeffs = _get_coefficients(gp.kernel)

	# comparing to DFM implementations
	U,V,ϕ,A=_init_matrices(kernel,x,yerr)
	logdetK = _factor_after_init!(A,U,V,ϕ)

	# evaluate the GP via cholesky factorization
	N = length(x)
    logdetK0 = _factorize!(gp.D, gp.U, gp.W, gp.ϕ, coeffs , x, gp.Σy)

	@test isapprox(logdetK,logdetK0)
	@test isapprox(gp.U,U)
	@test isapprox(gp.ϕ,ϕ)
	@test isapprox(gp.D,A)
	@test isapprox(gp.W,V)

	logL=logpdf(gp,y)

	# reconstuct the covariance matrix from decomposition
	Kmatrix = _reconstruct_K(gp,x)
	covariance = _full_solve(gp.kernel, x,yerr)

	@test isapprox(Kmatrix,covariance)

	LogL0 = -0.5 * (log(det(covariance)) + N * log(2pi) + (y' * inv(covariance)* y))
	@test isapprox(LogL0,logL)
	
	# compute the posterior GP, implied by observations
	# p_fx = posterior(gp,y)

	# compute the marginalized logLikelihood
	# logpdf(p_fx(x),y)

	# generate random samples from prior implied by GP 
	noise = randn(N)
	y0 = _sample_gp(gp,noise)
	

	# compute the conditional distribution (i.e. the predicted y* conditioned on observing y at inpute coordinates x)
	function full_math(gp::CeleriteGP,y::AbstractVector,x::AbstractVector)
        # μ = _k_matrix(gp,gp.x,x_train)' * inv(_k_matrix(gp,gp.x,gp.x)  + gp.Σy)  * (y_train )#.- mean(gp.x)) .+ mean(x)  
		# C = _k_matrix(gp,x_train,x_train) - _k_matrix(gp,gp.x,x_train)' * (_k_matrix(gp,gp.x,gp.x)  + gp.Σy) *  _k_matrix(gp,gp.x,x_train)

		μ =  diag(kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,x)) + gp.Σy) .* (y )#.- mean(gp.x))
		C = kernelmatrix(gp.kernel,x,x)  - kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,gp.x) + gp.Σy) * kernelmatrix(gp.kernel,gp.x,x)
        σ² =  diag(C)
        return μ,σ² 
    end

	@time "Choleksy decomposition" chol_y,chol_var=mean_and_var(gp,y,true_x)
	
	α = apply_inverse(gp,y)
	# M = N*4
    # tpred = sort!(rand(M)) .* 200
	@time "Ambikasaran method" ypred = predict(gp.kernel, x, y, true_x, α) # can replace true_x with tpred
	@test maximum(abs.(ypred .- chol_y)) <= 1e-5

	# @time "Full matrix inversion" math_y, math_var = full_math(gp,y,x,true_x)
	# @test maximum(abs.(ypred .- math_y)) <= 1e-5

	# ypred == mean(p_fx,x)
	# orig_kernel=celerite.SHOTerm(log(0.1), log(2.0), log(0.5))
	# orig_gp=celerite.Celerite(orig_kernel)
	# orig_coeffs = celerite.get_all_coefficients(orig_gp.kernel)
	# celerite.compute_ldlt!(orig_gp, x, yerr) 

	# @test isapprox(gp.D, orig_gp.D)
	# @test isapprox(gp.W, orig_gp.W)
	# @test isapprox(gp.U, orig_gp.up)
	# @test isapprox(gp.phi,orig_gp.phi)

	# apply the inverse of the covariance matrix to a vector 
	# orig_alpha = celerite.apply_inverse_ldlt(orig_gp,y)
	# @test isapprox(alpha,orig_alpha)
	# orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
	# orig_y0 = celerite.simulate_gp_ldlt(orig_gp,noise)
	# @test isapprox(y0,orig_y0)
	# orig_ypred = celerite.predict_ldlt!(orig_gp,x,y,tpred)
	# @test isapprox(ypred,orig_ypred)
end
