using Pkg
Pkg.activate("../../.")
using BSON:@load
using Statistics, StatsBase
include("../../src/SNM.jl")
include("../../src/MCMC.jl")
include("../../src/Analyze.jl")
include("Auctionlib.jl")
include("montecarlo.jl")


function AuctionWrapper()
    nParams = size(PriorSupport()[1],1)
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)    
    chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info, verbosity=false, nthreads=4)
    vcat(θhat, Analyze(chain))
end

# the monitoring function
function AuctionMonitor(sofar, results)
    if mod(sofar,10) == 0
        theta = TrueParameters()
        nParams = size(theta,1)
        println("__________ replication: ", sofar, "_______________")
        println("Results so far")
        println("parameter estimates")
        dstats(results[1:sofar,1:nParams]; short=true)
        println("CI coverage")
        clabels = ["99%","95%","90%"]
        prettyprint(reshape(mean(results[1:sofar,nParams+1:end],dims=1),nParams,3),clabels)
    end
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    reps = 100   # desired number of MC reps
    n_returns = 8 # 
    pooled = 1  # do this many reps b
    montecarlo(AuctionWrapper, AuctionMonitor, comm, reps, n_returns, pooled)
    MPI.Finalize()
end

main()





