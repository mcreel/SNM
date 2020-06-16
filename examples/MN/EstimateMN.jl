include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("MNlib.jl")
using BSON:@load

function EstimateMN()
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(auxstat(TrueParameters()), NNmodel, transform_stats_info)    
    @time chain, θhat = MCMC(m, NNmodel, transform_stats_info)
    #npdensity(chain[:,1])
end
EstimateMN()


