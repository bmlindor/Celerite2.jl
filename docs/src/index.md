# Celerite2.jl Documentation

```@meta
CurrentModule = Celerite2
```

```@docs 
RealKernel(log_a,log_c)
```
```@docs
Complex(log_a,log_b,log_c,log_d)
```

```@docs
SHOKernel(log_S,log_Q,log_ω0)
```

```@docs
RotationKernel(σ,period,Q0,dQ,frac)
```

```@docs
CeleriteGP(k::Tk,x::AbstractVector, σ::AbstractVector) where Tk <: CeleriteKernel
```

```@docs
logpdf(gp,y)
```

```@docs
rand(gp,N)
```

```@docs
posterior(gp,y)
```

```@docs
mean(gp_posterior,x)
```