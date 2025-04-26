## Testing

# include("../src/cel2.jl")
include("../deprecated/celerite.jl/src/celerite.jl")
using .Celerite2,Random,Statistics,Test,LinearAlgebra,Optim,StatsBase,Distributions

include("test_kernels.jl")
include("test_gp.jl")