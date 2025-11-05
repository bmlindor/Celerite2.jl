module Celerite2 

using Random, Base, LinearAlgebra, Statistics, StatsBase
using KernelFunctions, Distributions, AbstractGPs
using OMEinsum, Polynomials, ForwardDiff

import AbstractGPs: MeanFunction 
import Base.+, Base.*, Base.length, Base.product

export RealKernel, ComplexKernel, SHOKernel, CeleriteKernel
export RotationKernel, CeleriteKernelSum, CeleriteKernelProduct
export AbstractCeleriteGP, CeleriteGP, PosteriorCeleriteGP
export get_kernel, set_kernel!
export _init_matrices
# export posterior,
include("kernels.jl")
include("core.jl") # core math for cholesky factorization, applying inverse, matrix multiplication
include("psd.jl")
include("gp.jl") 
include("utils.jl")
# include("grad.jl")
end 
