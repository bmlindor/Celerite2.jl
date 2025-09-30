An updated julia v1+ implementation of [celerite](https://celerite.readthedocs.io/en/) (DFM et al 2017) and [celerite2](https://celerite2.readthedocs.io/en/latest/) built for use with [JuliaGPs](https://github.com/JuliaGaussianProcesses) . Provides consistent results with the celerite python module

## Notes: 
Still need to construct celerite2 kernels as KernelFunctions

- diff kernel (see [TermDiff](https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermDiff) )
- convolution kernel (for exposure time; see [TermConvolution](https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermConvolution) )
- matern3/2 kernel
	
While TermDiff and TermConvolution exist in celerite2, you can't do kernel operations on a TermConvolution so it might not be necessary.  
