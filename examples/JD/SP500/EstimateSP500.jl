using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles, Econometrics
using BSON:@save
using BSON:@load
using DelimitedFiles
#=
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
rets = data[:,1]
RV = data[:,2]
BV = data[:,3]
z = auxstat(rets, RV, BV)
m = mean(min.(max.(Float64.(nnmodel(TransformStats(z, nninfo)')),model.lb),model.ub),dims=2)
# draw a chain of length 10000, and get the extremum estimator
chain, θhat = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true)
# save visualize results
writedlm("chain", chain)
writedlm("thetahat", θhat)
=#
chain = readdlm("chain")
chn = Chains(chain)
display(chn)
plot(chn)
savefig("chain.png")
savefig(npdensity(chain[:,1]), "mu.png")
savefig(npdensity(chain[:,2]), "kappa.png")
savefig(npdensity(chain[:,3]), "alpha.png")
savefig(npdensity(chain[:,4]), "sigma.png")
savefig(npdensity(chain[:,5]), "rho.png")
savefig(npdensity(chain[:,6]), "lambda0.png")
savefig(npdensity(chain[:,7]), "lambda1.png")
savefig(npdensity(chain[:,8]), "tau.png")
