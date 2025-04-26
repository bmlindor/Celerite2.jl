README.md
An implementation of celerite (DFM et al 2017, https://celerite.readthedocs.io/en/) and the updated celerite2 (https://celerite2.readthedocs.io/en/latest/) in Julia v1+

TODO:
[ ] construct celerite kernels as KernelFunctions
	[-] simple harmonic oscillator
	[-] real and complex
	[-] stellar rotation 
	[-] operations (celerite sum and product)
	[-] diff kernel (see https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermDiff)
	[ ] convolution kernel (for exposure time; see https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermConvolution) 
	[ ] matern3/2 kernel
	[-] retrieve and update kernel parameters
	[-] compute kernel matrix
Notes: 	each KernelFunctions.Kernel is length 1 so I overloaded Base.size to contain lengths of the kernel components.
		I think a convolution term is similar to a KernelFunctions.Transform with_lengthscale, but I could be wrong. While TermDiff and TermConvolution exist in celerite2, you can't do kernel operations on a TermConvolution so it might not be necessary. 

[-] core math 
	[-] cholesky factorization
	[-] applying inverse
	[-] matrix multiplication
	[-] sampling random noise 
[-] construct finite gaussian process (as multivariate normal distribution)
	[-] compute marginal logLikelihood
	[-] prior mean, variance
	[-] prior covariance
[ ] predicting future process trained on observations 
	[ ] construct PosteriorGP
	[-] posterior mean, variance
	[ ] posterior covariance
Notes: AbstractGPs is not recognizing CeleriteGP as a type alias for FiniteGP, per https://github.com/JuliaLang/julia/issues/40448 so not doing that. Also, JuliaGPs collab requires a posterior function that constructs a PosteriorGP, but I haven't done that. 
	There's an error in computing the covariance of a predictive process that also exists in celerite.jl that needs to be resolved.
[-] power spectrum density 
	[-] compute PSD
	[-] compute roots of PSD (sturm's theorem)
[ ] ensure agreement with existing celerite
	[-] test_kernels.jl  
	[-] test_gp.jl
	[ ] test_opt.jl
Notes: I've been comparing EA's celerite.jl results for the Getting Started page.
	Prior to optimizing, the results are the same to machine precision (i.e. < 1e-12 in test_gp.jl). So the math is correct. However, multiple issues appear while optimizing.
		EA uses the log of the kernel components (I believe to ensure that Optim can sample -Inf to Inf) but if I use the linear components (as per the python celerite) I have been getting an error that the diagonal matrix is not positive definite. In this test, my _factorize! function can't compute the logdeterminant of K, I added the NaNMath package to mitigate this but this instead results optimized values that aren't actually optimized.
		I added a check during the optimizing that applies sturm's theorem (which checks for positive definiteness but I get similar not-actually-optimized results. So I reverted back to log kernel components and that seemed to help.
		The optimized parameters in celerite.jl and Celerite2.jl do not always pass the isapprox() test. Not a good indicator of implementation accuracy due to random draw of observed values, but the absolute maximum differences are on the other of -6
		Also, the full math (equ 82 and 83) doesn't agree with the predicted posterior when plotted (Do I need to plot x vs mu_{math} instead of true_{x} vs mu_{math}?)
[ ] MCMC sample of posterior GP
	[ ] use Turing NUTS() to characterize uncertainties on the kernel parameters
