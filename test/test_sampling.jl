using HDF5
data = h5open("/simulated_gp_data.h5")

x=read(data["data"]["xobs"]);
y=read(data["data"]["yobs"]);
yerr= read(data["data"]["yerr"]);
true_x=read(data["data"]["true_x"]);
true_y=read(data["data"]["true_y"]);

using Turing, LogExpFunctions
@model function gp_model(x,yerr,y)
    # Assumptions (i.e. placing Priors on gp terms)
    prior_sigma = 2.0
    μ ~ Normal(0.0, prior_sigma)
    logS01 ~ Normal(0.0, prior_sigma)
    # logQ1 ~ Normal(0.0, prior_sigma)
    logω01 ~ Normal(0.0, prior_sigma)
    logS02 ~ Normal(0.0, prior_sigma)
    logQ2 ~ Normal(0.0, prior_sigma)
    logω02 ~ Normal(0.0, prior_sigma)
    logjitter ~ Normal(0.0,0.1)

    # define gp that represents a distribution over functions (separate from Turing)
    term1 = Celerite2.SHOKernel(softplus(logS01), log(0.25),softplus(logω01))
    term2 = Celerite2.SHOKernel(softplus(logS02),softplus(logQ2),softplus(logω02))
    kernel = term1 + term2
    # want gp ~  MVNormal ; but need to compute gp before we can sample it 
    gp = Celerite2.CeleriteGP(kernel,x,yerr.+exp(logjitter),μ)

    # want positive definite D vector, maybe need to edit core.jl to assert this 
    # if any(mean(gp) .< 0.0)
        # Turing.@addlogprob! -Inf
    # else
        Turing.@addlogprob! logpdf(gp,y) 
    # end
    # Observations (i.e. dependent variable)    
    # gps ~ product_distribution(gp) #stackoverflow error?
    # y ~ marginals(gp)
    # y ~ product_distribution(Normal.(μ,yerr .+ jitter)) # DO I NEED TO SAMPLE THIS?
    # μs = mean(posterior(gp,y),x); conditioning on y?
    # post_gp=posterior(gp,y) 
    y ~ MvNormal(mean(gp),gp.Σy)
    # mymu,myvar=Celerite2.mean_and_var(gp,y,x)
    # return 
    return (logS01=logS01, logω01=logω01 , logS02=logS02, logQ2=logQ2, logω02=logω02,logjitter=logjitter,y=y)
end

model_gp = gp_model(x,yerr,y) 
model_gp_y  = gp_model(x,yerr,y) | (; y); # conditioned on y

fit_y_test = sample(model_gp_y, Turing.NUTS(1_000, 0.4), MCMCThreads(), 2_000,1,progress=true)

using StatsPlots
StatsPlots.plot(fit_y_test)
#=
# python equivalent
import numpy as np
import matplotlib.pyplot as plt
import celerite2
from celerite2 import terms
import h5py 
data = h5py.File('research/multis/simulated_gp_data.h5','r')
y=np.array(data['data']['yobs'])
x=np.array(data['data']['xobs'])
yerr = np.array(data['data']['yerr'])
true_x = np.array(data['data']['true_x'])
true_y = np.array(data['data']['true_y'])
w0=3.0; Q=1.0/np.sqrt(2.0)
S0 = np.var(y) / (w0 * Q)
term1 = terms.SHOTerm(w0=w0, Q=Q,S0=S0)
w0=3.0; Q=1.0
S0 = np.var(y) / (w0 * Q)
term2 = terms.SHOTerm(w0=w0, Q=Q,S0=S0)
kernel= term1+term2
gp = celerite2.GaussianProcess(kernel, mean=0.0)
gp.compute(np.sort(x), yerr=yerr)
import pymc as pm
from celerite2.pymc import GaussianProcess, terms as pm_terms
prior_sigma = 2.0
with pm.Model() as model:
    mean = pm.Normal("mean", mu=0.0, sigma=prior_sigma)
    log_w1 = pm.Normal("log_w1", mu=0.0, sigma=prior_sigma)
    log_S1 = pm.Normal("log_S1", mu=0.0, sigma=prior_sigma)
    term1 = pm_terms.SHOTerm(w0=pm.math.exp(log_w1),Q=0.25,S0=pm.math.exp(log_S1))
    log_w2 = pm.Normal("log_w2", mu=0.0, sigma=prior_sigma)
    log_S2 = pm.Normal("log_S2", mu=0.0, sigma=prior_sigma)
    log_Q2 = pm.Normal("log_Q2", mu=0.0, sigma=prior_sigma)
    term2 = pm_terms.SHOTerm(w0=pm.math.exp(log_w2), S0=pm.math.exp(log_S2), Q=pm.math.exp(log_Q2))
    log_jitter = pm.Normal("log_jitter", mu=0.0, sigma=prior_sigma)
    kernel =  term1 + term2
    gp = GaussianProcess(kernel, mean=mean)
    gp.compute(x, diag=yerr**2 + pm.math.exp(log_jitter), quiet=True)
    gp.marginal("obs", observed=y)
    trace = pm.sample(tune=1000,draws=1000,target_accept=0.95,init="adapt_full",cores=2,chains=2,random_seed=34923)
# 
# Initializing NUTS using adapt_full...
# /opt/anaconda3/lib/python3.12/site-packages/pymc/step_methods/hmc/quadpotential.py:760: UserWarning: QuadPotentialFullAdapt is an experimental feature
#   warnings.warn("QuadPotentialFullAdapt is an experimental feature")
# Multiprocess sampling (2 chains in 2 jobs)
NUTS: [mean, log_jitter, log_w1, log_Q1, log_S1, log_w2, log_S2]
                                                                                                           
  Progress            Draws   Divergences   Step size   Grad evals   Sampling Speed   Elapsed   Remaining  
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━   2000    0             0.27        15           103.42 draws/s   0:00:19   0:00:00    
  ━━━━━━━━━━━━━━━━━   2000    0             0.36        15           109.04 draws/s   0:00:18   0:00:00    
                                                                                                           
Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 19 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
>>> pm.summary(trace)
             mean     sd  hdi_3%  hdi_97%  ...  mcse_sd  ess_bulk  ess_tail  r_hat
mean        0.032  0.754  -1.436    1.423  ...    0.028    1104.0     966.0    1.0
log_w1     -0.828  0.647  -2.026    0.316  ...    0.015    1225.0    1484.0    1.0
log_S1      1.919  1.333  -0.566    4.301  ...    0.036     967.0     909.0    1.0
log_w2      1.094  0.086   0.924    1.237  ...    0.006    1086.0     463.0    1.0
log_S2     -3.397  0.734  -4.668   -2.071  ...    0.045     915.0     468.0    1.0
log_Q2      1.868  0.802   0.365    3.392  ...    0.024     823.0     616.0    1.0
log_jitter -5.885  0.807  -7.332   -4.542  ...    0.031    1136.0     643.0    1.0


trace.posterior["log_jitter"][i] ]
def set_mc_params(params, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel =  terms.SHOTerm(S0=theta[0], Q=0.70710,w0=theta[1]) + terms.SHOTerm(S0=theta[2], Q=theta[3], w0=theta[4]) 
    gp.compute(x, diag=yerr**2 + theta[5], quiet=True)
    return gp
for i in range(0,100):
    kernel =  terms.SHOTerm(S0=np.exp(trace.posterior["log_S1"][0,i]), Q=0.70710,w0=np.exp(trace.posterior["log_w1"][0,i])) + terms.SHOTerm(S0=np.exp(trace.posterior["log_S2"][0,i]), Q=np.exp(trace.posterior["log_Q2"][0,i]), w0=np.exp(trace.posterior["log_w2"][0,i])) 
    gp = celerite2.GaussianProcess(kernel, mean=trace.posterior["mean"][0,i])
    gp.compute(x, diag=yerr**2 + np.exp(trace.posterior["log_jitter"][0,i]), quiet=True)
    conditional = gp.condition(y, true_x)
    plt.plot(true_x, conditional.sample(), color="C0", alpha=0.1)

    gp.condition(y,true_x)

with model:
    post_gp = pm.sample_posterior(100)
=#