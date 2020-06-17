# example of using trained net to do a single
# Baysian estimation using MCMC

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
writedlm("chain", chain)
savefig(npdensity(chain[:,1]), "MNp1.png")
savefig(npdensity(chain[:,2]), "MNp2.png")
savefig(npdensity(chain[:,3]), "MNp3.png")
savefig(npdensity(chain[:,4]), "MNp4.png")
savefig(npdensity(chain[:,5]), "MNp5.png")


