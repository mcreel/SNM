project="MN"  # set to one of the projects in examples: SV, DPD, ARMA, MN
run_title = "working" # Monte Carlo results written to this file
mcreps = 1000 # how many reps?

# load code
include("examples/"*project*"/"*project*"lib.jl")
include("src/SNM.jl")
include("src/MakeNeuralMoments.jl") # the specialized MCMC using net 
include("src/Analyze.jl") # computes the confidence intervals, etc
include("src/MCMC.jl") # the specialized MCMC using net 
using BSON:@load

function RunProject()

# generate the trained net: comment out when done for the chosen model
nParams = size(PriorSupport()[1],1)
TrainingTestingSize = Int64(nParams*2*1e3) # 20,000 training and testing for each parameter
MakeNeuralMoments(auxstat, TrainingTestingSize) # already done for the 4 examples

# Monte Carlo study of confidence interval coverage for chosen model
results = zeros(mcreps,4*nParams)
# load the trained net: note, there are trained nets in the dirs of each project,
# to use those, edit the following line to set the correct path
@load "neural_moments.bson" NNmodel transform_stats_info
for mcrep = 1:mcreps
    # generate a draw of neural moments at true params
    m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)    
    @show m
    @time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info)
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
end
RunProject()


