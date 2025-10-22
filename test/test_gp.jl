@testset "gp" begin
	kernel=SHOKernel(log(0.1),log(2.0), log(0.5))
	gp=CeleriteGP(kernel,x,yerr)
	# @test fx isa AbstractGPs.FiniteGP

	coeffs = _get_coefficients(gp.kernel)

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

	alpha = apply_inverse(gp,y)
	# logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
	logL0=logpdf(gp,y)
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
	# @test isapprox(orig_logL,logL)

	p_fx = posterior(gp,y)

	# compute the marginalized logLikelihood

	logpdf(p_fx(x),y)

	# generate random samples from prior implied by GP 
	# noise = randn(N)
	# orig_y0 = celerite.simulate_gp_ldlt(orig_gp,noise)
	# y0 = _sample_gp(gp,noise)
	# dist()
	# @test isapprox(y0,orig_y0)
	# _reconstruct_K(gp,x)
	# compute the conditional distribution (i.e. the predicted y* conditioned on observing y at inpute coordinates x)
    # M = N*4
    # tpred = sort!(rand(M)) .* 200
	# orig_ypred = celerite.predict_ldlt!(orig_gp,x,y,tpred)
	# ypred = predict(gp, x, y, tpred)
	# @test isapprox(ypred,orig_ypred)
end
