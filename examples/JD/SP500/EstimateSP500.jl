using Pkg
Pkg.activate("../../../")
include("../../../src/SNM.jl")
include("../MCMC.jl")
include("JDlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig

function main()
# generate the trained net: comment out when done for the chosen model
nParams = size(PriorSupport()[1],1)
# load SP500 data
data = readdlm("SP500.txt")
rets = data[2:end,2]
RV = data[2:end,3]
BV = data[2:end,4]
@load "neural_moments.bson" NNmodel transform_stats_info
# make neural stats
lb, ub = PriorSupport()
z = auxstat(rets, RV, BV) # stats for SP500 data
m = min.(max.(Float64.(NNmodel(TransformStats(z, info)')),lb),ub)
@time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info; verbosity=true, nthreads=10, rt=0.1)
return chain, θhat
end
chain, θhat = main()
writedlm("chain", chain)
writedlm("thetahat", θhat)
savefig(npdensity(chain[:,1]), "mu.svg")
savefig(npdensity(chain[:,2]), "kappa.svg")
savefig(npdensity(chain[:,3]), "alpha.svg")
savefig(npdensity(chain[:,4]), "sigma.svg")
savefig(npdensity(chain[:,5]), "rho.svg")
savefig(npdensity(chain[:,6]), "lambda0.svg")
savefig(npdensity(chain[:,7]), "lambda1.svg")
savefig(npdensity(chain[:,8]), "tau.svg")
