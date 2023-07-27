using SimulatedNeuralMoments
using Flux, MCMCChains
using StatsPlots, Distributions
using DelimitedFiles, LinearAlgebra
using BSON:@save
using BSON:@load
using CSV, DataFrames
include("JDlib.jl")

function EstimateJD(TrainTestSize=1, Epochs=1000)

# generate some data, and get sample size 
df = CSV.read("spy.csv", DataFrame);
y = (Matrix(df[:, [:rets, :rv, :bv]]))
n = size(y,1)

# fill in the structure that defines the model
lb, ub = PriorSupport() # bounds of support
model = SNMmodel("SPY JD", n, lb, ub, GoodData, InSupport, Prior, PriorDraw, auxstat)
# train the net, and save it and the transformation info
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
θnn = NeuralMoments(auxstat(y), model, nnmodel, nninfo)[:]
@show θnn
# settings
names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ","lnL"]
S = 100
covreps = 1000
tuninglength = 2000
finallength = 20000
burnin = 100
verbosity = 10 # show results every X dras
tuning = 0.1
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
    Σp = cov(chain[:,1:end-2])
    acceptance < 0.2 ? tuning =0.75 : nothing
    acceptance > 0.3 ? tuning =1.5 : nothing
    proposal(θ) = rand(MvNormal(θ, tuning*Σp))
    start = vec(mean(chain[:,1:end-2],dims=1))
    chain = mcmc(start, tuninglength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
    acceptance = mean(chain[:,end])
end

# final long chain
start = vec(mean(chain[:,1:end-2],dims=1))
chain = mcmc(start, finallength, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)

# get the summary info
acceptance = mean(chain[:,end])
println("acceptance rate: $acceptance")
chain = Chains(chain[:,1:end-1], names) # convert to Chains type, drop acc. rate
display(chain)
display(plot(chain))
savefig("JDchain.png")
@save  "spy_MCMC_results.bson" chain tuning Σp acceptance
end

