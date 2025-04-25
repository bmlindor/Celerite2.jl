## Testing

include("../src/cel2.jl")
include("../../src/celerite.jl")
using .Celerite2,Random,Statistics,Test,LinearAlgebra,Optim,StatsBase

# include("test_kernels.jl")
include("test_gp.jl")