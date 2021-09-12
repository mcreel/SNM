using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles, Optim
using BSON:@save
using BSON:@load
using DelimitedFiles
include("JDlib.jl")
lb, ub = PriorSupport()
# fill in the structure that defines the model
model = SNMmodel("SP500 estimation", lb, ub, InSupport, Prior, PriorDraw, auxstat)
# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model)
@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
data = readdlm("sp500.txt")
m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
@show m
m = min.(max.(lb,m),ub)
@show m
chain, P, tuning = MCMC(m, 10000, model, nnmodel, nninfo, verbosity=true, do_cue = true, tuningloops=2, nthreads=10, burnin=40, tuning=0.75)
# save visualize results
writedlm("chain", chain)
writedlm("P", P)
writedlm("tuning", tuning)

