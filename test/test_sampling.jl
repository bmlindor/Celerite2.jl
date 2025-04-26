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
