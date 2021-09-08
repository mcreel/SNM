#=
#using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles, Optim, CSV, DataFrames
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
data = readdlm("sp500.txt")
m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
@show m
m = min.(max.(lb,m),ub)
@show m

#chain, P, tuning = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true, do_cue = true, tuningloops=2, nthreads=10, burnin=40, tuning=0.75)
# save visualize results
writedlm("chain", chain)
writedlm("P", P)
writedlm("tuning", tuning)
=#
chain = readdlm("chain")
chn = Chains(chain)
display(chn)
plot(chn)
savefig("chain.png")

# do this part by hand, using saved chain, to eliminate dependency on Econometrics
savefig(npdensity(chain[:,1]), "mu.png")
savefig(npdensity(chain[:,2]), "kappa.png")
savefig(npdensity(chain[:,3]), "alpha.png")
savefig(npdensity(chain[:,4]), "sigma.png")
savefig(npdensity(chain[:,5]), "rho.png")
savefig(npdensity(chain[:,6]), "lambda0.png")
savefig(npdensity(chain[:,7]), "lambda1.png")
savefig(npdensity(chain[:,8]), "tau.png")

