# example of using trained net to do a single
# Baysian estimation using MCMC
using Pkg
Pkg.activate("../../")
include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("MNlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig
using Statistics
function Estimate()
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
    @time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info; verbosity=false)
    chain, θhat
end
chain, θhat = Estimate()
writedlm("chain", chain)
savefig(npdensity(chain[:,1]), "MNp1.png")
savefig(npdensity(chain[:,2]), "MNp2.png")
savefig(npdensity(chain[:,3]), "MNp3.png")
savefig(npdensity(chain[:,4]), "MNp4.png")
savefig(npdensity(chain[:,5]), "MNp5.png")

