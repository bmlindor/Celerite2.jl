## Testing


using Celerite2,Random,Statistics,Test,LinearAlgebra,StatsBase
using Distributions,DelimitedFiles,Optim
import AbstractGPs: posterior
import Celerite2: _get_coefficients, _sample_gp
import Celerite2: predict, apply_inverse, _k_matrix, _reconstruct_K
import Celerite2: _factorize! , _solve!
import Celerite2: _init_matrices, _factor_after_init!

filename = string("simulated_gp_data.txt")
data=readdlm(filename,comments=true)
x = data[:,1];
y = data[:,2];
yerr = data[:,3];
true_x = data[:,4];
true_y = data[:,5];

include("test_kernels.jl")
include("test_opt.jl")
# include("test_gp.jl")
#= function comp_gp(μ,variance)
		clf()
		ax = subplot(111)
		# ax.plot(true_x, μ_ldlt, "r",label=string("Opt. celerite.jl kernel;"," runtime:",round(telapsed_ldlt,sigdigits=3)))
		ax.plot(true_x,μ , label=string("Opt. Celerite2.jl;"," runtime:",round(telapsed,sigdigits=3)),ls=":");
		ax.plot(true_x,μ_math , label=string("full math;"," runtime:",round(telapsed_math,sigdigits=3)),ls="-.");
		ax.plot(true_x,μ_rec , "g",label="recursive prediction, no variance",ls="--");
		# ax.plot(true_x, true_y, "k", lw=1.5, alpha=0.3)
		ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
		ax.fill_between(true_x, μ_ldlt.+sqrt.(variance_ldlt), μ_ldlt.-sqrt.(variance_ldlt), color="orange", alpha=0.3)
		ax.fill_between(true_x, μ.+sqrt.(variance), μ.-sqrt.(variance), color="blue", alpha=0.3)
		ax.legend(loc="upper left")
		ax.set_ylim(-2.5,2.5);ax.set_ylabel("y")
		ax.set_xlim(-1,11);ax.set_xlabel("x")
		ax.set_title("Julia implementations.")
		savefig("test/figures/comparing.png")
    end

	# comp_gp(myμ,myvar)

	freq = range(1.0 / 8,stop= 1.0 / 0.3, length=500)
	omega = 2 * pi * freq

    function plot_psd(gp)
    	clf()
    	ax=subplot()
		ax.plot(freq,Celerite2._psd(gp.kernel.kernels[1],omega),label="term 1")
		ax.plot(freq,Celerite2._psd(gp.kernel.kernels[2],omega),label="term 2")
		ax.plot(freq,Celerite2._psd(gp.kernel,omega),ls="--",color="b",label="optimized Celerite2.jl")
		ax.set_xlabel("frequency [1 / day]")
		ax.set_ylabel("Power [day ppt²]")
		ax.set_title("maximum likelihood psd")
		ax.legend()
		ax.set_xlim(minimum(freq),maximum(freq))
		ax.set_xscale("log")
		ax.set_yscale("log")
		savefig("test/figures/2025_psd.png")
    end
    # plot_psd(gp)
=#