# example of using trained net to do a single
# Baysian estimation using MCMC
using Pkg
Pkg.activate("../../")
include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("SVlib.jl")
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
savefig(npdensity(chain[:,1]), "phi.png")
savefig(npdensity(chain[:,2]), "rho.png")
savefig(npdensity(chain[:,3]), "sig.png")

