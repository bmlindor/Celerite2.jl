include("../src/cel2.jl")
# include("../../src/celerite.jl")
using .Celerite2,Random,Statistics,Test,LinearAlgebra,AbstractGPs,Optim,StatsBase,Turing,PyPlot

# @testset "optimize_gp" begin
	rng=MersenneTwister(42)
	x = sort(cat(1, 3.8 .* rand(57), 5.5 .+ 4.5 .* rand(68);dims=1));
	yerr = 0.08 .+ (0.22-0.08) .*rand(length(x));
	y = 0.2.*(x.-5.0) .+ sin.(3.0.*x .+ 0.1.*(x.-5.0).^2) .+ yerr .* randn(length(x));

	true_x=collect(range(0,stop=10,length=126))
	true_y = 0.2 .*(true_x.-5) .+ sin.(3 .*true_x .+ 0.1.*(true_x.-5).^2);

	# non periodic component
	Q = 1.0/sqrt(2.0) ; w0 = 3.0
	S0 = var(y) ./ (w0 * Q)
	comp_1=Celerite2.SHOKernel(log(S0),log(Q),log(w0))

	# periodic component
	Q = 1.0; w0 = 3.0
	S0 = var(y) ./ (w0 * Q)
	comp_2=Celerite2.SHOKernel(log(S0),log(Q),log(w0))

	kernel = comp_1 + comp_2
	gp=Celerite2.CeleriteGP(kernel,x,yerr)

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

	# orig_coeffs = celerite.get_all_coefficients(orig_gp.kernel)
	# coeffs = Celerite2._get_coefficients(gp.kernel)
	@show gp.kernel
	# println("diff in coeffs:",orig_coeffs," vs ", coeffs)
	logL=logpdf(gp,y)
	orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
	println("Difference in logLikelihood: ",orig_logL-logL)
	# @testset "initial GP" begin
	# @test isapprox(orig_logL,logL)
	# @test isapprox(gp.D, orig_gp.D)
	# @test isapprox(gp.W,orig_gp.W)
	# @test isapprox(gp.U,orig_gp.up)
	# @test isapprox(gp.phi,orig_gp.phi)
	# end

	# matrixK = Celerite2._k_matrix(gp,true_x,gp.x)
	# orig_matrixK = celerite.get_matrix(orig_gp,true_x,orig_gp.x)
	# println("Diff. in K matrices: ",max(matrixK .- orig_matrixK))
	logcomp_1=Celerite2.logSHOKernel(log(S0),log(Q),log(w0))
	logcomp_2=Celerite2.logSHOKernel(log(S0),log(Q),log(w0))
	log_kernel = logcomp_1 + logcomp_2
	log_gp = Celerite2.CeleriteGP(log_kernel,x,yerr)
	logvector = Celerite2.get_kernel(log_kernel)
	## optimize GP 	#  maximize the (log) marginal likelihood wrt. hyperparameters
	vector = Celerite2.get_kernel(kernel)
	mask = ones(Bool,size(kernel))
	mask[2] = false 	# We don't want to fit the first Q

	function nll(params)
		vector[mask] = params
		# build gp prior
		Celerite2.set_kernel!(gp.kernel,vector)
		# build finite gp
		gp_trial=Celerite2.CeleriteGP(gp.kernel,x,yerr)
		        # sturms theorem
       coeffs = Celerite2._get_coefficients(gp.kernel)

        if Celerite2._check_pos_def(coeffs)
        			# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
		else
			println("bad")
            return -Inf
       end
        # else
		# end
	end
		res_NM = optimize(nll, vector[mask],Optim.NelderMead())
	println("minimized NM results:",res_NM.minimizer)
	res_LBFGS = optimize(nll, vector[mask],Optim.LBFGS())
	println("minimized LBFGS results:",res_LBFGS.minimizer)
	vector[mask] = res_LBFGS.minimizer
	Celerite2.set_kernel!(gp.kernel,vector)
	logL=logpdf(gp,y)
	function other_min()

	function logL_wrapperlog(params)
		logvector[mask] = params
		# build gp prior
		Celerite2.set_kernel!(log_kernel,logvector)
		# build finite gp
		gp_trial=Celerite2.CeleriteGP(log_kernel,x,yerr)
		# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
	end

	function nll_mean(params)
		vector_mean[mask_mean] = params
		# build gp prior
		Celerite2.set_kernel!(log_kernel,vector_mean[2:end])
		# build finite gp
		gp_trial=Celerite2.CeleriteGP(log_kernel,x,yerr,vector_mean[1])
		# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
	end
	vector_mean = [0.0;Celerite2.get_kernel(kernel)]
	mask_mean = ones(Bool,length(vector_mean))
	mask_mean[3] = false
	# res_mean_LBFGS = optimize(nll_mean, vector_mean[mask_mean],Optim.LBFGS())
	logres_LBFGS = optimize(logL_wrapperlog, logvector[mask],Optim.LBFGS())
	# @time res_auto = optimize(logL_wrapper, init_vector[mask],BFGS();autodiff = :forward)
	logvector[mask] = logres_LBFGS.minimizer
	Celerite2.set_kernel!(log_gp.kernel,logvector)
	loglogL=logpdf(log_gp,y)
	end

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
		println("minimized celerite LBFGS results:",orig_res.minimizer)
	orig_vector[mask] = Optim.minimizer(orig_res)
	celerite.set_parameter_vector!(orig_gp.kernel, orig_vector)
	orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
	logL=logpdf(gp,y)
	println("Difference in logLikelihood: ",orig_logL-logL)
	@testset "optimized gp" begin
	# @test isapprox(orig_logL,logL)
	# @test maximum(abs.(exp.(orig_vector) - vector)) < 1e-5
	# alpha = Celerite2._solve!(gp.D, gp.U, gp.W, gp.phi, y)
	# orig_alpha = celerite.apply_inverse_ldlt(orig_gp,y)
	# @test maximum(abs.(alpha - orig_alpha)) < 1e-12
	# @test maximum(abs.(log_gp.D- orig_gp.D)) < 1e-12 
	# @test isapprox(log_gp.D, orig_gp.D)
	# # @test maximum(abs.(log_gp.W- orig_gp.W)) < 1e-12 
	# @test isapprox(log_gp.W,orig_gp.W)
	# # @test maximum(abs.(log_gp.U- orig_gp.up)) < 1e-12
	# @test isapprox(log_gp.U,orig_gp.up)
	end

	# Celerite2.set_kernel!(gp.kernel,vector)
	t1 = time()
	mu_ldlt, variance_ldlt = celerite.predict_full_ldlt(orig_gp, y, true_x, return_var=true)
	telapsed_ldlt = time() -t1

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

	function myposterior(gp,x_train,y_train,x,full_math::Bool=true)
        alpha = Celerite2.apply_inverse(gp,y_train)
        Kxs = Celerite2._k_matrix(gp,x,x_train)
        mu1 = Kxs * alpha
        C1_tmp = Celerite2._k_matrix(gp,x_train)
        KxsT = transpose(Kxs)
        C1 = C1_tmp .- Kxs * Celerite2.apply_inverse(gp,KxsT)
        # σ²1 = diag(C1)
        if !full_math
         v=zeros(length(x_train))
        #  for i=1:length(x_train)
        #  v[i] = - sum(KxsT[:,i] .* Celerite2.apply_inverse(gp,KxsT[:,i]),1)
	    # end
	    σ²1 =  Celerite2._get_value(gp.kernel,[0.0])[1] .+ v
        # or 
        	return mu1, σ²1[1,:]
        end
        if full_math
        mu2 = mean(x) .+ Celerite2._k_matrix(gp,gp.x,x_train)' * inv(Celerite2._k_matrix(gp,gp.x,gp.x)  + gp.Σy)  * (y_train .- mean(gp.x))
        C = Celerite2._k_matrix(gp,x_train,x_train) - Celerite2._k_matrix(gp,gp.x,x_train)' * (Celerite2._k_matrix(gp,gp.x,gp.x)  + gp.Σy) *  Celerite2._k_matrix(gp,gp.x,x_train)
        σ²2 =  diag(C)
        return mu2,σ²2 
	    end
    end

    function plot_gp(mu,variance)
    clf()
    ax = subplot(111)
    ax.plot(true_x, mu_ldlt, "r",label=string("Optimized celerite kernel;","elapsed runtime:",round(telapsed_ldlt,sigdigits=3)))
	ax.plot(x,orig_mu_rec , "g",label="celerite prediction");
    # ax.plot(x,full_mu , label=string("full_Math;","elapsed runtime:",round(telapsed_full,sigdigits=3)));
    ax.plot(x,mu , label=string("mymu;","elapsed runtime:",round(telapsed_equ,sigdigits=3)));
	# ax.plot(true_x,mu_rec , "g",label="recursive prediction, no variance");
	# ax.plot(x,mu_rec , "purple",label="test prediction");
    ax.plot(true_x, true_y, "k", lw=1.5, alpha=0.3)
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    ax.fill_between(true_x, mu_ldlt.+sqrt.(variance_ldlt), mu_ldlt.-sqrt.(variance_ldlt), color="orange", alpha=0.3)
    # ax.fill_between(x, full_mu+sqrt.(full_var), full_mu-sqrt.(full_var), color="cornflowerblue", alpha=0.3)
    # ax.fill_between(x, mu1.+sqrt.(variance1), mu1.-sqrt.(variance1), color="purple", alpha=0.3)
    ax.fill_between(x, mu.+sqrt.(variance), mu.-sqrt.(variance), color="purple", alpha=0.3)
    ax.legend()
    ax.set_ylim(-3,3);ax.set_ylabel("y")
	ax.set_xlim(0,10);ax.set_xlabel("x")
	ax.set_title("Julia implementations.")
    end
     mu_rec = Celerite2.predict!(gp,x,y,true_x)
     orig_mu_rec = celerite.predict_ldlt!(orig_gp,true_x,y,x)
     # mu1,variance1=test_posterior(gp,true_x,true_y,x)
	plot_gp(mymu,0.0)

	function threshold_density(lower, upper)
	    σ = @. abs(upper - lower)/3.0
	    μ = @. (upper + lower) / 2
	    return @. Normal(μ, σ)
	end

	function prior_gp(theta)
		# dists = (
	    #     threshold_density, # Mass ratio
	    #     threshold_density, # Period
	    #     threshold_density, # initial transit time
	    #     threshold_density, # ecosϖ
	    #     threshold_density, # esinϖ
		# 	threshold_density, # transit depth
		# )

	    # mass = (1e-5,1e-4)
	    # Ps = (
	    #     (1.50, 1.52),
	    #     (2.41,2.43))
	    # t0s = (
	    #     (7257.5, 7257.6),
	    #     (7258.55, 7258.61))
	    # ecs = (-0.1,0.1)
	    # ess = (-0.1,0.1)
	    # δs = (0.005,0.008)
		# q_n = (0.0,1.0)
		# rstar = (0.0005,0.0006)

		# prior = []
	end

	freq = range(1.0 / 8,stop= 1.0 / 0.3, length=500)
	omega = 2 * pi * freq


    function _test_psd(kernel::Celerite2.CeleriteKernel, ω)
        ar, cr, ac, bc, cc, dc = Celerite2._get_coefficients(kernel)
        ω² = ω.^2 
        ω⁴ = ω².^2 
        p = zeros(length(ω²))
        for i in 1:length(ar)
            p = p + ar[i]*cr[i] ./ (cr[i]*cr[i] .+ ω²)
        end
        for i in 1:length(ac)
            ω0² = cc[i]*cc[i]+dc[i]*dc[i]
            p = p .+ ((ac[i]*cc[i].+bc[i]*dc[i])*ω0².+(ac[i]*cc[i]-bc[i]*dc[i]).*ω²) ./ (ω⁴ + 2.0*(cc[i]*cc[i]-dc[i]*dc[i]).*ω².+ω0²*ω0²)
        end
        return sqrt(2.0 / pi) .* p
    end
    _test_psd(gp.kernel,omega)

    function plot_psd(gp)
    	ax=subplot()
		ax.plot(freq,_test_psd(gp.kernel.kernels[1],omega),label="term 1")
		ax.plot(freq,_test_psd(gp.kernel.kernels[2],omega),label="term 2")
		ax.plot(freq,_test_psd(gp.kernel,omega),ls="--",color="k",label="optimized kernels")
		ax.set_xlabel("frequency [1 / day]")
		ax.set_ylabel(L"Power [day ppt$^2$]")
		ax.set_title("maximum likelihood psd")
		ax.legend()
		ax.set_xlim(minimum(freq),maximum(freq))
		ax.set_xscale("log")
		ax.set_yscale("log")
    end
	@model function gp_model(xobs,yobs,eobs)
		prior_sigma = 2.0

		mean ~ Normal(0.0, prior_sigma)
		logS01 ~ Normal(0.0, prior_sigma)
		logQ1 ~ Normal(0.0, prior_sigma)
		logω01 ~ Normal(0.0, prior_sigma)
		logS02 ~ Normal(0.0, prior_sigma)
		logQ2 ~ Normal(0.0, prior_sigma)
		logω02 ~ Normal(0.0, prior_sigma)

		term1 = Celerite2.SHOKernel(exp(logS01),exp(logQ1),exp(logω01))
		term2 = Celerite2.SHOKernel(exp(logS02),exp(logQ2),exp(logω02))
		gp_trial = Celerite2.CeleriteGP(term1+term2,xobs,eobs, mean)
		# Celerite2.compute!(gp_trial,xobs,ebs )
		Celerite2.logpdf(gp_trial,yobs)
		Celerite2._get_psd(gp_trial.kernel,omega)
		# add the marginal likelihood to the Turing model?
		# fobs ~ MvNormal(flux, eobs)
	end

		
# end

	# Celerite2.set_kernel!(gp.kernel,exp.(vector))
	# println("diff in optimized params:",exp.(vector) - init_vector)
	# compute!()
	# orig_coeffs = celerite.get_all_coefficients(orig_gp.kernel)
	# coeffs = Celerite2._get_coefficients(gp.kernel)
	# println("diff in coeffs:",orig_coeffs," vs ", coeffs)

	# post_gp = posterior(gp,y)
	# mu = Celerite2.mymean(post_gp,true_x)
	# println("Max diff in posterior mu:",maximum(abs.(mu - mu_ldlt)))

	# orig_ypred_ldlt = celerite.predict_ldlt!(orig_gp, x, y, true_x[1:57])
    # # tpred2 = zeros(tpred)
	# ypred = Celerite2.predict!(gp, x, y, true_x[1:57])
	# println("Max diff in predicted times:",maximum(abs.(ypred - orig_ypred_ldlt)))
	# orig_alpha = celerite.apply_inverse_ldlt(orig_gp,y)
	# alpha = Celerite2.apply_inverse(gp,y)
	# println("max diff in alpha :",maximum(abs.(orig_alpha - alpha)))

	# U0,V0,phi0,A0=Celerite2._init_matrices(gp.kernel,x,yerr)


	# @test (phi0 - orig_gp.phi) <1e-6


    # ax.plot(x,mean(x) .+ diag(kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,x)) .* (y .- mean(gp.x))) , ls = "--",label="full_Math",zorder = 1);

    # @test isapprox(full_Math_mu)
    # return orig_vector,vector
# end
