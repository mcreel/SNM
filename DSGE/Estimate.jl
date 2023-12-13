using SimulatedNeuralMoments, Flux, SolveDSGE, MCMCChains
using Distributions, StatsPlots, DelimitedFiles
using BSON:@save
using BSON:@load

# get the things to define the structure for the model
include("CKlib.jl") # contains the functions for the DSGE model
function main()
lb, ub = PriorSupport()

# draw a sample at the design parameters, from the prior, or use the official "real" data
#data = CKdgp(TrueParameters(), dsge, 1)[1]
data = readdlm("dsgedata.txt")
n = size(data,1)

# fill in the structure that defines the model
model = SNMmodel("DSGE example", n, lb, ub, InSupport, Prior, PriorDraw, auxstat)

# Here, you can train the net from scratch, or use a previous run
# train the net, and save it and the transformation info
#nnmodel, nninfo = MakeNeuralMoments(model)
#@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# define the neural moments using the data
θnn = NeuralMoments(auxstat(data), model, nnmodel, nninfo)[:]
@info "the raw NN estimates: " θnn

## define the proposal and the log-likelihood
S = 50 # number of simulation reps
covreps = 500 # for the proposal covariance
length = 10000
burnin = 500
verbosity = 100 # show results every X draws
tuning = 1.0
junk, Σp = mΣ(θnn, covreps, model, nnmodel, nninfo)
proposal(θ) = rand(MvNormal(θ, tuning*Σp))
lnL = θ -> snmobj(θ, θnn, S, model, nnmodel, nninfo)

## run a short chain to improve proposal
# tuning the chain and creating a good proposal may
# need care - this is just an example!
@info "running a short chain to allow refining of the tuning"
chain = mcmc(θnn, 1000, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

@info "refining the tuning, and running the final chain"
Σp = cov(chain[:,1:end-2])
acceptance = mean(chain[:,end])
acceptance < 0.2 ? tuning = 0.5 : nothing
acceptance > 0.3 ? tuning = 1.50 : nothing
proposal2(θ) = rand(MvNormal(θ, tuning*Σp))
# final chain using second round proposal
chain = mcmc(θnn, length, lnL, model, nnmodel, nninfo, proposal2, burnin, verbosity)

# visualize results
chn = Chains(chain, ["β", "γ", "ρ₁", "σ₁", "ρ₂", "σ₂", "nss"])
plot(chn)
savefig("chain.png")
display(chn)
end
main()
