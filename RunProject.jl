using Pkg
Pkg.activate(".")
using BSON:@load
project="MN"  # set to one of the projects in examples: SV, DPD, ARMA, MN
include("examples/"*project*"/"*project*"lib.jl")
include("src/SNM.jl")
include("src/MakeNeuralMoments.jl")
include("src/MCMC.jl")
include("src/Analyze.jl")

run_title = "working" # Monte Carlo results written to this file
mcreps = 1000 # how many reps?

# the monitoring function
function Monitor(sofar, results)
    if mod(sofar,10) == 0
        theta = TrueParameters()
        nParams = size(theta,1)
        println("__________ replication: ", sofar, "_______________")
        println("Results so far")
        println("parameter estimates")
        est = results[1:sofar,1:nParams]
        err = est .- TrueParameters()'
        b = mean(err, dims=1)
        s = std(err, dims=1)
        rmse = sqrt.(b.^2 + s.^2)
        prettyprint([b' rmse'], ["bias", "rmse"])
        dstats(est; short=true)
        println("CI coverage")
        clabels = ["99%","95%","90%"]
        prettyprint(reshape(mean(results[1:sofar,nParams+1:end],dims=1),nParams,3),clabels)
    end
end

function RunProject()
    # generate the trained net: comment out when done for the chosen model
    nParams = size(TrueParameters(),1)
    #TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each parameter
    # try larger training to see effect on MA model
    TrainingTestingSize = Int64(nParams*2*1e5) # 200,000 training and testing for each parameter
    MakeNeuralMoments(auxstat, TrainingTestingSize) # already done for the 4 examples
    # Monte Carlo study of confidence interval coverage for chosen model
    results = zeros(mcreps,4*nParams)
    # load the trained net: note, there are trained nets in the dirs of each project,
    # to use those, edit the following line to set the correct path
    @load "neural_moments.bson" NNmodel transform_stats_info
    Threads.@threads for mcrep = 1:mcreps
        # generate a draw of neural moments at true params
        m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)    
        @time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info, verbosity=false)
        r = vcat(θhat, Analyze(chain))
        println("current result: ", r)
        results[mcrep,:] = r 
        println("____________________________")
    end
    writedlm(run_title, results)
    Monitor(mcreps, results)
    return nothing
end
RunProject()

