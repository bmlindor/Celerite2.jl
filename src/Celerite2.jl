module Celerite2 

using Random, Base, LinearAlgebra, Statistics, StatsBase
using KernelFunctions, Distributions, AbstractGPs
using OMEinsum,Polynomials

import AbstractGPs: MeanFunction 
import Base.+, Base.*, Base.length, Base.product

export RealKernel, ComplexKernel, SHOKernel
export RotationKernel, CeleriteKernelSum, CeleriteKernelProduct
export CeleriteGP
export get_kernel, set_kernel!
export make_test_data

include("kernels.jl")
include("core.jl") # core math for cholesky factorization, applying inverse, matrix multiplication
include("psd.jl")
include("gp.jl") 
include("utils.jl")


end 
