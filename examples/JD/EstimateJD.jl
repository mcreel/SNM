using Pkg
Pkg.activate("../../")
include("../../src/SNM.jl")
include("MCMC.jl")
include("JDlib.jl")
using BSON:@load
using DelimitedFiles
using Plots:savefig
include("../../src/MakeNeuralMoments.jl") # the specialized MCMC using net 

function main()
# generate the trained net: comment out when done for the chosen model
nParams = size(PriorSupport()[1],1)
#TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each parameter
TrainingTestingSize = Int64(nParams*5e3) # 20,000 training and testing for each parameter
MakeNeuralMoments(auxstat, TrainingTestingSize) # already done for the 4 examples
#@load "neural_moments.bson" NNmodel transform_stats_info
#m = NeuralMoments(TrueParameters(), auxstat, 1, NNmodel, transform_stats_info)
#@time chain, θhat = MCMC(m, auxstat, NNmodel, transform_stats_info; verbosity=true, nthreads=10, rt=0.1)
#return chain, θhat
end
main()
#=
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
=#
