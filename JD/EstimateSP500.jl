using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles, Optim
using BSON:@save
using BSON:@load
using DelimitedFiles
# get the things to define the structure for the model
# For your own models, you will need to supply the functions
# found in MNlib.jl, using the same formats
include("JDlib.jl")
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("SP500 estimation", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
data = readdlm("SP500.txt")
data = Float64.(data[2:end,2:end])
m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
chain, P, tuning = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true, do_cue = true, tuningloops=2, nthreads=20, burnin=20, tuning=0.5)
# save visualize results
writedlm("chain", chain)
writedlm("P", P)
writedlm("tuning", tuning)
chn = Chains(chain)
display(chn)
plot(chn)
savefig("chain.png")
#= do this part by hand, using saved chain, to eliminate dependency on Econometrics
savefig(npdensity(chain[:,1]), "mu.png")
savefig(npdensity(chain[:,2]), "kappa.png")
savefig(npdensity(chain[:,3]), "alpha.png")
savefig(npdensity(chain[:,4]), "sigma.png")
savefig(npdensity(chain[:,5]), "rho.png")
savefig(npdensity(chain[:,6]), "lambda0.png")
savefig(npdensity(chain[:,7]), "lambda1.png")
savefig(npdensity(chain[:,8]), "tau.png")
=#
