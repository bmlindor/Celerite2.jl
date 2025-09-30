@testset "gp" begin
	Nobs=100
	N,x,yerr,y=Celerite2.make_test_data(Nobs)
	kernel=Celerite2.SHOKernel(log(0.1),log(2.0), log(0.5))
	gp=Celerite2.CeleriteGP(kernel,x,yerr)

	# @test fx isa AbstractGPs.FiniteGP

	orig_kernel=celerite.SHOTerm(log(0.1), log(2.0), log(0.5))
	orig_gp=celerite.Celerite(orig_kernel)

	coeffs = Celerite2._get_coefficients(gp.kernel)
	orig_coeffs = celerite.get_all_coefficients(orig_gp.kernel)

	# evaluate the GP via cholesky factorization
    logdetK = Celerite2._factorize!(gp.D, gp.U, gp.W, gp.phi, coeffs , x, gp.Σy)
	celerite.compute_ldlt!(orig_gp, x, yerr) 

	@test isapprox(gp.D, orig_gp.D)
	@test isapprox(gp.W, orig_gp.W)
	@test isapprox(gp.U, orig_gp.up)
	@test isapprox(gp.phi,orig_gp.phi)

	# apply the inverse of the covariance matrix to a vector 
	orig_alpha = celerite.apply_inverse_ldlt(orig_gp,y)
	alpha = Celerite2.apply_inverse(gp,y)
	@test isapprox(alpha,orig_alpha)

	# compute the marginalized logLikelihood
	logL=logpdf(gp,y)
	orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
	@test isapprox(orig_logL,logL)

	# generate random samples from prior implied by GP 
	noise = randn(Nobs)
	orig_y0 = celerite.simulate_gp_ldlt(orig_gp,noise)
	y0 = Celerite2._sample_gp(gp,noise)
	@test isapprox(y0,orig_y0)

	# compute the conditional distribution (i.e. the predicted y* conditioned on observing y at inpute coordinates x)
    M = N*4
    tpred = sort!(rand(M)) .* 200
	orig_ypred = celerite.predict_ldlt!(orig_gp,x,y,tpred)
	ypred = Celerite2.predict!(gp, x, y, tpred)
	@test isapprox(ypred,orig_ypred)
end
