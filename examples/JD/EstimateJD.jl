# example of using trained net to do a single
# Baysian estimation using MCMC

include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("JDlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig

function main()
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
    println("neural moments: ", m)
    @time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info; verbosity=true)
    chain, θhat
end
chain, θhat = main()
writedlm("chain", chain)
savefig(npdensity(chain[:,1]), "mu0.png")
savefig(npdensity(chain[:,2]), "mu1.png")
savefig(npdensity(chain[:,3]), "kappa.png")
savefig(npdensity(chain[:,4]), "alpha.png")
savefig(npdensity(chain[:,5]), "sigma.png")
savefig(npdensity(chain[:,6]), "rho.png")
savefig(npdensity(chain[:,7]), "lambda0.png")
savefig(npdensity(chain[:,8]), "lambda1.png")


