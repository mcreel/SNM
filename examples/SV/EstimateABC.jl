# example of using trained net to do a single
# Baysian estimation using MCMC
using Distributions, KissABC
include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("SVlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig


@load "neural_moments.bson" NNmodel transform_stats_info
m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
println("neural moments: ", m)

prior = Factored(
    Uniform(0.05, 2.0),
    Uniform(0.0, 0.999),
    Uniform(0.05, 1.0));

# estimate covariance
Σinv = inv(EstimateΣ(m, 100, auxstat, NNmodel, transform_stats_info))
D(θ) = -1.0*H(θ, m, 10, auxstat, NNmodel, transform_stats_info, Σinv)

approx_density = ApproxPosterior(prior, D, 0.0)
res = sample(
    approx_density,
    AIS(10),
    MCMCThreads(),
    1000,
    10,
    burnin = 100,
    ntransitions = 10,
    progress = false
)
@show res

