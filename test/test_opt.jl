include("../src/Celerite2.jl")
# include("../../src/celerite.jl")
using .Celerite2,Random,Statistics,Test,KernelFunctions,LinearAlgebra,Optim,StatsBase,PyPlot

@testset "optimize" begin
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

	logL=logpdf(gp,y)
	orig_logL = celerite.log_likelihood_ldlt(orig_gp, y)
	# println("Difference in logLikelihood: ",orig_logL-logL)

	# Maximize the (log) marginal likelihood wrt. hyperparameters
	vector = Celerite2.get_kernel(kernel)
	mask = ones(Bool,size(kernel))
	mask[2] = false 	# We don't want to fit the first Q

	function nll(params)
		vector[mask] = params
		# build gp prior
		Celerite2.set_kernel!(gp.kernel,vector)
		# build finite gp
		gp_trial=Celerite2.CeleriteGP(gp.kernel,x,yerr)
        coeffs = Celerite2._get_coefficients(gp.kernel)
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
	Celerite2.set_kernel!(gp.kernel,vector)
	logL=logpdf(gp,y)
	# @show gp.kernel

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

	function nll_mean(params)
		vector_mean[mask_mean] = params
		# build gp prior
		Celerite2.set_kernel!(gp.kernel,vector_mean[2:end])
		# build finite gp
		gp_trial=Celerite2.CeleriteGP(gp.kernel,x,yerr,vector_mean[1])
		# compute log_marginal likelihood
		logL_trial=logpdf(gp_trial,y)
		return -logL_trial
	end
	vector_mean = [0.0;Celerite2.get_kernel(kernel)]
	mask_mean = ones(Bool,length(vector_mean))
	mask_mean[3] = false
	res_mean_LBFGS = optimize(nll_mean, vector_mean[mask_mean],Optim.LBFGS())
	t1 = time()
	mu_ldlt, variance_ldlt = celerite.predict_full_ldlt(orig_gp, y, true_x, return_var=true)
	telapsed_ldlt = time() -t1

    function full_math(gp::CeleriteGP,y_train::AbstractVector,x_train::AbstractVector,x::AbstractVector)
        mu = Celerite2._k_matrix(gp,gp.x,x_train)' * inv(Celerite2._k_matrix(gp,gp.x,gp.x)  + gp.Σy)  * (y_train )#.- mean(gp.x)) .+ mean(x)  
        C = Celerite2._k_matrix(gp,x_train,x_train) - Celerite2._k_matrix(gp,gp.x,x_train)' * (Celerite2._k_matrix(gp,gp.x,gp.x)  + gp.Σy) *  Celerite2._k_matrix(gp,gp.x,x_train)
        σ² =  diag(C)
        return mu,σ² 
    end
    t2=time()
	mymu,myvar=Celerite2.mean_and_var(gp,y,true_x)
	telapsed = time() - t2
    t3=time()
	mu_math, var_math = full_math(gp,y,x,true_x)
	telapsed_math = time() - t3
    function comp_gp(mu,variance)
    clf()
    ax = subplot(111)
    ax.plot(true_x, mu_ldlt, "r",label=string("Opt. celerite.jl kernel;"," runtime:",round(telapsed_ldlt,sigdigits=3)))
    ax.plot(true_x,mu , label=string("Opt. Celerite2.jl;"," runtime:",round(telapsed,sigdigits=3)),ls=":");
    ax.plot(true_x,mu_math , label=string("full math;"," runtime:",round(telapsed_math,sigdigits=3)),ls="-.");
	ax.plot(true_x,mu_rec , "g",label="recursive prediction, no variance",ls="--");
    # ax.plot(true_x, true_y, "k", lw=1.5, alpha=0.3)
    ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    ax.fill_between(true_x, mu_ldlt.+sqrt.(variance_ldlt), mu_ldlt.-sqrt.(variance_ldlt), color="orange", alpha=0.3)
    ax.fill_between(true_x, mu.+sqrt.(variance), mu.-sqrt.(variance), color="blue", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(-2.5,2.5);ax.set_ylabel("y")
	ax.set_xlim(-1,11);ax.set_xlabel("x")
	ax.set_title("Julia implementations.")
	savefig("test/figures/comparing.png")
    end
    mu_rec = Celerite2.predict!(gp,x,y,true_x)

	comp_gp(mymu,myvar)

	freq = range(1.0 / 8,stop= 1.0 / 0.3, length=500)
	omega = 2 * pi * freq

    function plot_psd(gp)
    	clf()
    	ax=subplot()
		ax.plot(freq,Celerite2._psd(gp.kernel.kernels[1],omega),label="term 1")
		ax.plot(freq,Celerite2._psd(gp.kernel.kernels[2],omega),label="term 2")
		ax.plot(freq,Celerite2._psd(gp.kernel,omega),ls="--",color="b",label="optimized Celerite2.jl")
		ax.set_xlabel("frequency [1 / day]")
		ax.set_ylabel(L"Power [day ppt$^2$]")
		ax.set_title("maximum likelihood psd")
		ax.legend()
		ax.set_xlim(minimum(freq),maximum(freq))
		ax.set_xscale("log")
		ax.set_yscale("log")
		savefig("test/figures/2025_psd.png")
    end
    plot_psd(gp)
    close()
end
