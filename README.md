# Celerite2
Scalable 1D Gaussian Processes in Julia version 1+ , first computed in  [![arxiv](https://img.shields.io/badge/arXiv-1703.09710-brightgreen)](https://arxiv.org/abs/1703.09710)

This is an updated implementation of [celerite](https://celerite.readthedocs.io/en/) and [celerite2](https://celerite2.readthedocs.io/en/latest/) — built with the [JuliaGPs](https://github.com/JuliaGaussianProcesses) ecosystem — that provides consistent results with the celerite python module.

## Installation
You can install the registered Celerite2 repo as a Julia package with the Julia `Pkg` manager.
```julia
using Pkg ; Pkg.add("Celerite2.jl")
```

## Example
```julia
using Random,Celerite2

rng=MersenneTwister(42)

# Generate synthetic data for sorted times `x`
x = sort(cat(1, 3.8 .* rand(57), 5.5 .+ 4.5 .* rand(68);dims=1));
yerr = 0.08 .+ (0.22-0.08) .*rand(length(x));
y = 0.2.*(x.-5.0) .+ sin.(3.0.*x .+ 0.1.*(x.-5.0).^2) .+ yerr .* randn(length(x));

# Define GP prior with Simple Harmonic Oscillator kernels:
# non-periodic component
Q = 1.0/sqrt(2.0)
 w0 = 3.0
S0 = var(y) ./ (w0 * Q)
non_per = Celerite2.SHOKernel(log(S0),log(Q),log(w0))

# periodic component
Q = 1.0
w0 = 3.0
S0 = var(y) ./ (w0 * Q)
per = Celerite2.SHOKernel(log(S0),log(Q),log(w0))

kernel = non_per + per
f = Celerite2.CeleriteGP(kernel,x,yerr)

# Compute the log-likelihood of the observations `y`
logpdf(f,y)

# Create a posterior GP trained at `y`
p_fx = posterior(gp,y)

# Compute the log-likelihood of the trained posterior
logpdf(p_fx(x),y)

# Efficiently predict the mean of the posterior GP `p_fx`
μ = mean(p_fx,x)
```

## Notes: 
Still need to construct celerite2 kernels as KernelFunctions

- diff kernel (see [TermDiff](https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermDiff) )
- convolution kernel (for exposure time; see [TermConvolution](https://celerite2.readthedocs.io/en/latest/_modules/celerite2/terms/#TermConvolution) )
- matern3/2 kernel
	
While TermDiff and TermConvolution exist in celerite2, you can't do kernel operations on a TermConvolution so it might not be necessary.  

Please open an issue if you would like to contribute.
