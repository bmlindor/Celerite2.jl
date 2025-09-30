
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
