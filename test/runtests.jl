## Testing


using Celerite2,Random,Statistics,Test,LinearAlgebra,StatsBase
using Distributions,DelimitedFiles,Optim
using Polynomials
import AbstractGPs: posterior
import Celerite2: _get_coefficients, _sample_gp, _check_pos_roots
import Celerite2: predict, apply_inverse, _k_matrix, _reconstruct_K
import Celerite2: _factorize! , _solve!, _full_solve
import Celerite2: _init_matrices, _factor_after_init!

using ForwardDiff
filename = string("simulated_gp_data.txt")
data=readdlm(filename,comments=true)
x = data[:,1];
y = data[:,2];
yerr = data[:,3];

# truth = readdlm("sinusoidal_gp_data.txt",comments=true)
true_x = data[:,4];
true_y = data[:,5];

# include("test_kernels.jl")
include("test_gp.jl")
# include("test_opt.jl")