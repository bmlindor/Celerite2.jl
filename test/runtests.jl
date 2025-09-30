## Testing

include("../src/Celerite2.jl")
using Pkg
# Pkg.add(path="https://github.com/ericagol/celerite.jl/",subdir="src/")
using .Celerite2,Random,Statistics,Test,LinearAlgebra,Optim,StatsBase,Distributions

# include("test_kernels.jl")
# include("test_gp.jl")
