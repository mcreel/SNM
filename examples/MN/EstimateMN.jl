include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("MNlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig

function EstimateMN()
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(TrueParameters(), NNmodel, transform_stats_info)    
    @time chain, θhat = MCMC(m, NNmodel, transform_stats_info)
    chain, θhat
end
chain, θhat = EstimateMN()
savefig(npdensity(chain[:,1]), "param1.svg")
savefig(npdensity(chain[:,2]), "param2.svg")
savefig(npdensity(chain[:,3]), "param3.svg")
savefig(npdensity(chain[:,4]), "param4.svg")
savefig(npdensity(chain[:,5]), "param5.svg")


