README.md
Implementation of celerite (DFM et al 2017) in Julia and updated celerite2.
TODO:
[-] construct celerite kernels KernelFunctions, where Kernels are length 1
	[-] simple harmonic oscillator
	[-] complex
	[-] real
	[-] stellar rotation 
	[-] celerite sum
	[-] celerite product

[ ] methods 
	[-] retrieve and update kernel parameters
	[-] compute kernel matrix

[ ] core math 
	[-] cholesky factorization
	[-] applying inverse
	[-] matrix multiplication
	[-] sampling random noise 

[-] construct finite gaussian process (multivariate normal distribution)
	[-] compute marginal logLikelihood
	[-] prior mean, variance
	[-] prior covariance

[ ] predicting future process trained on observations 
		[] posterior mean, variance
		[] posterior covariance

[ ] power spectrum density 
	[-] compute PSD
	[-] compute roots of PSD (sturm's theorem)

[ ] ensure agreement with existing celerite
	[-] test_kernels.jl  
	[-] test_gp.jl
	[ ] test_opt.jl

