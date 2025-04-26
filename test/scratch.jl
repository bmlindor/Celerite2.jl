
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