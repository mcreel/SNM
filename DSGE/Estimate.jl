#= 

This is written to be used interactively, with VScode, to explain the
methods, step by step. For good performance, it is better to wrap 
everything into a function. See the example in the JD directory for
how to do that.

Start julia as "julia --proj -t auto" to use threads

=#

using SimulatedNeuralMoments, Flux, SolveDSGE, MCMCChains
using Distributions, StatsPlots, DelimitedFiles, PrettyTables
using BSON:@save
using BSON:@load

## get the things to define the structure for the model
include("CKlib.jl") # contains the functions for the DSGE model

## DATA: draw a sample at the design parameters, from the prior, or use the official "real" data
# Use this for a random true parameter vector
θtrue = PriorDraw()
data = dgp(PriorDraw(), dsge, 1)[1]
# use this for the "official" sample used in the notes
# θtrue = TrueParameters()
# data = readdlm("dsgedata.txt")

## fill in the structure that defines the model
n = size(data,1)
lb, ub = PriorSupport()
model = SNMmodel("DSGE example", n, lb, ub, GoodData, InSupport, Prior, PriorDraw, auxstat)

#= 
# UNCOMMENT this block to see training
# train the net, and save it and the transformation info
TrainTestSize = 10000
Epochs = 1000
nnmodel, nninfo, params, stats, transf_stats = MakeNeuralMoments(model, TrainTestSize=TrainTestSize, Epochs=Epochs)
# @save "neuralmodel.bson" nnmodel nninfo # use this line to save the trained neural net 
# examine the transformed stats to ensure that outliers
# have been controlled. We want to see some distance between the whiskers.
@info "checking the transformed statistics for outliers"
for i = 1:size(transf_stats,2)
    boxplot(transf_stats[:,i],title="statistic $i")
    savefig("stat$i.png")
end
=#

## load the pre-trained net
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

## define the neural moments using the data
θnn = NeuralMoments(auxstat(data), model, nnmodel, nninfo)[:]
pretty_table([θtrue θnn],header = (["θtrue", "θnn"]))
# settings
S = 100
covreps = 1000
tuninglength = 500
finallength = 1000
burnin = 100
verbosity = 100 # show results every X draws
tuning = 0.5

## define the proposal and the log-likelihood
junk, Σp = mΣ(θnn, covreps, model, nnmodel, nninfo)
while !isposdef(Σp)
    for i = 1:size(Σp,1)
        Σp[i,i] += 1e-5
    end
end    
proposal(θ) = rand(MvNormal(θ, tuning*Σp))
lnL = θ -> snmobj(θ, θnn, S, model, nnmodel, nninfo)

## run a short chain to improve proposal
# tuning the chain and creating a good proposal may
# need care - this is just an example!
chain = mcmc(θnn, tuninglength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
acceptance = mean(chain[:,end])

## update proposal until acceptance rate is good
while acceptance < 0.2 || acceptance > 0.3
    global tuning, chain, acceptance, start
    acceptance < 0.2 ? tuning *= 0.75 : nothing
    acceptance > 0.3 ? tuning *= 1.5 : nothing
    proposal(θ) = rand(MvNormal(θ, tuning*Σp))
    start = vec(mean(chain[:,1:end-2],dims=1))
    chain = mcmc(start, tuninglength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
    acceptance = mean(chain[:,end])
end

## final long chain
start = vec(mean(chain[:,1:end-2],dims=1))
chain = mcmc(start, finallength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

## visualize results
chn = Chains(chain[:,1:end-2], ["β", "γ", "ρ₁", "σ₁", "ρ₂", "σ₂", "nss"])
plot(chn)
savefig("chain.png")
display(chn)
pretty_table([θtrue θnn mean(chain[:,1:end-2],dims=1)[:]], header = (["θtrue", "θnn", "θmcmc"]))
@save "MCMC_results.bson" chain tuning Σp acceptance
