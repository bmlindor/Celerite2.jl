@testset "gp" begin
	Nobs=100
	N,x,yerr,y=Celerite2.make_test_data(Nobs)
	kernel=Celerite2.SHOKernel(0.1,2.0, 0.5)
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
	@test isapprox(gp.W,orig_gp.W)
	@test isapprox(gp.U,orig_gp.up)
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

# @testset "predict_gp" begin
# 	Nobs=10
# 	N,x,yerr,y=Celerite2.make_test_data(Nobs)
# 	kernel=Celerite2.SHOKernel(0.1,2.0, 0.5)
# 	gp=Celerite2.CeleriteGP(kernel,x,yerr)

# 	orig_kernel=celerite.SHOTerm(log(0.1), log(2.0), log(0.5))
# 	orig_gp=celerite.Celerite(orig_kernel)
# 	orig_coeffs = celerite.get_all_coefficients(orig_gp.kernel)
# 	celerite.compute_ldlt!(orig_gp, x, yerr) 
# end

	# k=Celerite2._k_matrix(gp,x)
	# K=Celerite2.kernelmatrix(gp.kernel,x,yerr)
	# @test isapprox(yerr.^2,(K - k))
	# @test maximum(abs.(K - (diagm(yerr.^2) .+ k))) < 1e-12
	# orig_gp.D,orig_gp.W,orig_gp.up,orig_gp.phi = celerite.init_matrices!(orig_coeffs..., x, yerr, orig_gp.W, orig_gp.phi, orig_gp.up, orig_gp.D)
# # expct, U,V = celerite.init_matrices(coeffs..., x, yvar)

# 	# 	# BL: other names for these variables:  D, X, U, phi
	# @test isapprox(orig_gp.up,gp.U)
	# @test isapprox(orig_gp.phi,gp.phi)
# 	# @test  maximum(gp.up' .- U0) < 1e-12
# 	# @test gp.phi' == phi0
	# # 	# @test gp.Σy==diagm(yerr)
# # 	# @test gp.x==x
# # 	# @test A0 ==  diag(fx.Σy).^2 .+ (sum(ac) + sum(ar))

# 	# A,W,U,new_phi=Celerite2.DFM_factor!(U,phi,D,W)
# 	# @test gp.up' == U0
# # 	# return maximum(gp.up' .- U)
# # 		# @test 
# # 		return gp.W' - W
# # 	# res=Celerite2.GP(D2)(x)
# 	# return gp.W' .- V0
# return #alpha,orig_alpha
# end
