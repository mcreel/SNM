# example of using trained net to do a single
# Baysian estimation using MCMC
using Distributions, KissABC
include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("JDlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig


@load "neural_moments.bson" NNmodel transform_stats_info
m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
println("neural moments: ", m)

prior = Factored(
    Uniform(-0.1, 0.1),
    Uniform(-0.1, 0.1),
    Uniform(0.0, 0.5),
    Uniform(-1.0, 3.0),
    Uniform(0.01, 3.0),
    Uniform(-0.999, 0.0),
    Uniform(0.0, 0.05),
    Uniform(3.0, 5.0));

# method with identity weight
D(θ) = -1.0*H(θ, m, 10, auxstat, NNmodel, transform_stats_info, Matrix(1.0I, size(θ,1), size(θ,1)))

approx_density = ApproxPosterior(prior, D, 0.1)
res = sample(
    approx_density,
    AIS(50),
    MCMCThreads(),
    1000,
    4,
    burnin = 300,
    ntransitions = 10,
    progress = false,
)
@show res

