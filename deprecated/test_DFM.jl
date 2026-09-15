# include("../src/grad.jl")
using DelimitedFiles,Statistics,Distributions,Celerite2
filename = string("research/Celerite2.jl/test/simulated_gp_data.txt")
data=readdlm(filename,comments=true)
x = data[:,1];
y = data[:,2];
yerr = data[:,3];
Q = 1.0/sqrt(2.0) ; w0 = 3.0
S0 = var(y) ./ (w0 * Q)
comp_1=Celerite2.SHOKernel(log(S0),log(Q),log(w0))
# U,V,ϕ,A=get_matrices(comp_1,x,yerr)
gp_DFM = DFMCeleriteGP.InitCeleriteGP(comp_1,x,yerr)
@time "find logL " logL = DFMCeleriteGP.loglike(gp_DFM,y)
println("DFM: ",logL) 
#  a, U, V, P = Celerite2.get_matrices(term, x, sigma.^2 .+ zeros(N))
#   Celerite2.factor!(U, P, a, V)
#   Celerite2.
gp=Celerite2.CeleriteGP(comp_1,x,yerr)
@time "find logL0" logL0=logpdf(gp,y)

println("Current: ",logL0)

@show maximum(abs.(gp.W - gp_DFM.W'))
@show maximum(abs.(gp.U - gp_DFM.U'))
@show maximum(abs.(gp.ϕ - gp_DFM.ϕ'))
@show maximum(abs.(gp.D .-gp_DFM.D))


gp_DFM2= DFMCeleriteGP.InitCeleriteGP(comp_1,x,yerr)
@time "find grad logL " logL = DFMCeleriteGP.grad_loglike(gp_DFM2,y)

true_x = data[:,4];
true_y = data[:,5];

function gp_loglikelihood(x,y)
    function loglikelihood(params)
        logS01, logQ1, logω01 = softplus(params[1]),softplus(params[2]),softplus(params[3])
        term1 = Celerite2.SHOKernel(logS01, logQ1, logω01)
        logS02, logQ2, logω02 = softplus(params[4]),softplus(params[5]),softplus(params[6])

        term2 = Celerite2.SHOKernel(logS02,logQ2,logω02)
        kernel = term1 + term2
        μ = params[8]
        logjitter = softplus(params[7])
        gp = Celerite2.CeleriteGP(kernel,x,yerr.+exp(logjitter),μ)
        return logpdf(gp,y)
    end
    return loglikelihood
end

loglik_train = gp_loglikelihood(x_train, y_train)

logprior(params) = logpdf(MvNormal(Eye(8)), params)

struct LogJointTrain 
    dim::Int
end

# Log joint density
LogDensityProblems.logdensity(problem::LogJointTrain, θ) = loglik_train(θ) + logprior(θ)

# The parameter space is two-dimensional
LogDensityProblems.dimension(problem::LogJointTrain) = problem.dim

# `LogJointTrain` does not allow to evaluate derivatives of the log density function
function LogDensityProblems.capabilities(problem::Type{LogJointTrain})
    return LogDensityProblems.LogDensityOrder{0}()
end

rng = Random.MersenneTwister(0)
n_samples, n_adapts = 100, 10

# compute the derivatives of the log joint density with automatic differentiation.
# define an Hamiltonian system of the log joint probability.
logjoint_train = ADgradient(Val(:ForwardDiff), LogJointTrain())
metric = DiagEuclideanMetric()
hamiltonian = Hamiltonian(metric, logjoint_train)

# define a leapfrog solver, with initial step size chosen heuristically.
initial_params = rand()
initial_ϵ = find_good_stepsize(hamiltonian, initial_params)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler
proposal = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(;max_depth=3)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
samples, stats = sample(
    hamiltonian, proposal, initial_params, n_samples, adaptor, n_adapts; progress=false
)
# return samples,stats
# end