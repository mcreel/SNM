#using Pkg
#Pkg.activate("../")

run_title = "working"
mcreps = 1000

# This code only estimates using the GMM form, with the
# neural net. The code in release v1.0 estimates
# the alternative versions discussed in the working
# paper.
# 
# Also, this uses the extemum estimator
# to compute an estimate of the covariance
# of the moments, which is fixed over the 
# MCMC iterations, like a 2 step GMM esimator,
# rather than a CUE estimator. This is faster.

#_____________________ Choose the model _____________________#
#include("ARMA/ARMAlib.jl")
#include("DPD/DPDlib.jl")
#include("SV/SVlib.jl")
include("MN/MNlib.jl")
#_____________________ Choose the model _____________________#


# this makes the simulated data, trains net,
# and saves all information needed to compute
# the neural moments
include("src/MakeNeuralMoments.jl")
# computes the confidence intervals, etc
include("src/Analyze.jl")
# the specialized MCMC using net 
include("src/MCMC.jl")
using BSON:@load

function RunProject()
lb, ub = PriorSupport()
nParams = size(lb,1)
TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each parameter
# generate the raw training data
MakeNeuralMoments(auxstat, TrainingTestingSize)
#=
results = zeros(mcreps,4*nParams)
@load "neural_moments.bson" NNmodel transform_stats_info
for mcrep = 1:mcreps
    # generate a draw at true params
    m = auxstat(TrueParameters())    
    @time chain, θhat = MCMC(m, NNmodel, transform_stats_info)
    results[mcrep,:] = vcat(θhat, Analyze(chain))
    println("__________ replication: ", mcrep, "_______________")
    println("Results so far")
    println("parameter estimates")
    dstats(results[1:mcrep,1:nParams]; short=true)
    println("CI coverage")
    clabels = ["99%","95%","90%"]
    prettyprint(reshape(mean(results[1:mcrep,nParams+1:end],dims=1),nParams,3),clabels)
    println("____________________________")
end
writedlm(run_title, results)
=#
end
RunProject()


