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
model = SNMmodel("DSGE example", n, lb, ub, GoodData, InSupport, Prior, PriorDraw, auxstat)


# train the net, and save it and the transformation info
TrainTestSize = 100000
Epochs = 1000
#nnmodel, nninfo, params, stats, transf_stats = MakeNeuralMoments(model, TrainTestSize=TrainTestSize, Epochs=Epochs)
#@save "neuralmodel.bson" nnmodel nninfo params stats transf_stats # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo transf_stats # use this to load a trained net


## examine the transformed stats to ensure that outliers
# have been controlled. We want to see some distance between the whiskers.
@info "checking the transformed statistics for outliers"
for i = 1:size(transf_stats,2)
    boxplot(transf_stats[:,i],title="statistic $i")
    savefig("stat$i.png")
end

# define the neural moments using the data
θnn = NeuralMoments(auxstat(data), model, nnmodel, nninfo)[:]
@show θnn
# settings
S = 100
covreps = 1000
tuninglength = 2000
finallength = 20000
burnin = 100
verbosity = 100 # show results every X draws
tuning = 0.5

# define the proposal
junk, Σp = mΣ(θnn, covreps, model, nnmodel, nninfo)
while !isposdef(Σp)
    for i = 1:size(Σp,1)
        Σp[i,i] += 1e-5
    end
end    
proposal(θ) = rand(MvNormal(θ, tuning*Σp))
# define the logL
lnL = θ -> snmobj(θ, θnn, S, model, nnmodel, nninfo)

# run a short chain to improve proposal
# tuning the chain and creating a good proposal may
# need care - this is just an example!
chain = mcmc(θnn, tuninglength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
acceptance = mean(chain[:,end])

# update proposal until acceptance rate is good
while acceptance < 0.2 || acceptance > 0.3
    acceptance < 0.2 ? tuning *= 0.75 : nothing
    acceptance > 0.3 ? tuning *= 1.5 : nothing
    proposal(θ) = rand(MvNormal(θ, tuning*Σp))
    start = vec(mean(chain[:,1:end-2],dims=1))
    chain = mcmc(start, tuninglength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
    acceptance = mean(chain[:,end])
end

# final long chain
start = vec(mean(chain[:,1:end-2],dims=1))
chain = mcmc(start, finallength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

# visualize results
chn = Chains(chain[:,1:end-2], ["β", "γ", "ρ₁", "σ₁", "ρ₂", "σ₂", "nss"])
plot(chn)
savefig("chain.png")
display(chn)
@save "MCMC_results.bson" chain tuning Σp acceptance
end
main()
