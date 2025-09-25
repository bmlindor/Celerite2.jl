An implementation of celerite (DFM et al 2017, https://celerite.readthedocs.io/en/) and the updated celerite2 (https://celerite2.readthedocs.io/en/latest/) in Julia v1.
For the same simulated dataset, I find results which are consistent with the celerite.
##  Tasks:
	[ ] construct celerite2 kernels as KernelFunctions
		[ ] diff kernel (see https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermDiff)
		[ ] convolution kernel (for exposure time; see https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermConvolution) 
		[ ] matern3/2 kernel
## Notes: 	
- I think a convolution term is similar to a KernelFunctions.Transform with_lengthscale, but I could be wrong. While TermDiff and TermConvolution exist in celerite2, you can't do kernel operations on a TermConvolution so it might not be necessary. 

- AbstractGPs is not recognizing CeleriteGP as a type alias for FiniteGP, per https://github.com/JuliaLang/julia/issues/40448 so not doing that. 

- There's an error in computing the covariance of a predictive process that also exists in celerite.jl that needs to be resolved.

- I've been comparing EA's celerite.jl results for the Getting Started page. Prior to optimizing, the results are the same to machine precision (i.e. < 1e-12 in test_gp.jl). However, the optimized parameters in celerite.jl and Celerite2.jl do not always pass the isapprox() test. Not a good indicator of implementation accuracy due to random draw of observed values, but the absolute maximum differences are on the other of -6.  
