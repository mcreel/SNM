# example of using trained net to do a single
# Baysian estimation using MCMC
using Distributions, KissABC
include("../../src/SNM.jl")
include("../../src/Analyze.jl")
include("MNlib.jl")
using BSON:@load

function main()
    @load "neural_moments.bson" NNmodel transform_stats_info
    prior = Factored(
        Uniform(0.0, 3.0),
        Uniform(-2.0, 2.0),
        Uniform(0.0, 1.0),
        Uniform(0.0, 4.0),
        Uniform(0.0, 1.0))
    Reps = 100
    nParams = 5
    results = zeros(Reps,4*nParams)
    S = 10
    for rep = 1:Reps    
        m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
        Σinv = inv((1.0+1/S).*EstimateΣ(m, 100, auxstat, NNmodel, transform_stats_info))
        D(θ) = -1.0*H(θ, m, S, auxstat, NNmodel, transform_stats_info, Σinv)
        approx_density = ApproxPosterior(prior, D, 0.0)
        res = sample(
            approx_density,
            AIS(200),
            MCMCThreads(),
            1250,
            8,
            burnin = 500,
            ntransitions = 50,
            progress = false
        )
        chain = res[:,:,1]
        for i in 2:size(res,3)
            chain =[chain; res[:,:,i]]
        end    
        results[rep,:] = vcat(vec(mean(chain,dims=1)), Analyze(chain))
        println("__________ replication: ", rep, "_______________")
        println("Results so far")
        println("parameter estimates")
        dstats(results[1:rep,1:nParams]; short=true)
        println("CI coverage")
        clabels = ["99%","95%","90%"]
        prettyprint(reshape(mean(results[1:rep,nParams+1:end],dims=1),nParams,3),clabels)
        println("____________________________")
    end    
    return nothing
end
main()

