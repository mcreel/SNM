using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
include("SVlib.jl") # contains the functions for the DSGE model
function main()
lb, ub = PriorSupport()

# fill in the structure that defines the model
model = SNMmodel("SV example", lb, ub, InSupport, Prior, PriorDraw, auxstat)

# Here, you can train the net from scratch, or use a previous run
# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# draw a sample at the design parameters, from the prior, or use the official "real" data
data = SVmodel(TrueParameters())

# define the neural moments using the data
m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
# Here, you can create a new chain, or use the results from a previous run
# draw a chain of length 10000 plus 500 burnin
chain, P, tuning = MCMC(m, 1500, model, nnmodel, nninfo, covreps = 100, verbosity=true, do_cue = true)
chain = chain[501:end,:]
#writedlm("chain.txt", chain)
#writedlm("P", P)
#writedlm("tuning", tuning)
#chain = readdlm("chain.txt")

# visualize results
chn = Chains(chain, ["ϕ", "ρ", "σ"])
plot(chn)
#savefig("chain.png")
display(chn)
end
main()
