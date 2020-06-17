#using Pkg
#Pkg.activate("../")
#using SNM
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
#include("examples/ARMA/ARMAlib.jl")
#include("examples(DPD/DPDlib.jl")
#include("examples/SV/SVlib.jl")
include("examples/MN/MNlib.jl")


# this makes the simulated data, trains net,
# and saves all information needed to compute
# the neural moments
include("src/SNM.jl")
include("src/MakeNeuralMoments.jl") # the specialized MCMC using net 
include("src/Analyze.jl") # computes the confidence intervals, etc
include("src/MCMC.jl") # the specialized MCMC using net 
using BSON:@load

function RunProject()
lb, ub = PriorSupport()
nParams = size(lb,1)
# generate the trained net
TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each parameter
MakeNeuralMoments(auxstat, TrainingTestingSize) # already done for the 4 examples
results = zeros(mcreps,4*nParams)
#=
@load "neural_moments.bson" NNmodel transform_stats_info
for mcrep = 1:mcreps
    # generate a draw of neural moments at true params
    m = NeuralMoments(TrueParameters(), NNmodel, transform_stats_info)    
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


