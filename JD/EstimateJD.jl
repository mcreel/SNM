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
model = SNMmodel("SPY JD", n, lb, ub, InSupport, Prior, PriorDraw, auxstat)

# train the net, and save it and the transformation info
nnmodel, nninfo = MakeNeuralMoments(model, TrainTestSize=TrainTestSize, Epochs=Epochs)
@save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
@load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

# define the neural moments using the data
θnn = NeuralMoments(auxstat(y), model, nnmodel, nninfo)[:]
@show θnn
# settings
names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ"]
S = 50
covreps = 1000
length = 1000
burnin = 100
verbosity = 1 # show results every X draws
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
chain = mcmc(θnn, 1000, lnL, model, nnmodel, nninfo, proposal, burnin, verbosity)
Σp = cov(chain[:,1:end-2])
acceptance = mean(chain[:,end])
acceptance < 0.2 ? tuning =0.75 : nothing
acceptance > 0.3 ? tuning =1.5 : nothing
proposal2(θ) = rand(MvNormal(θ, tuning*Σp))

# final chain using second round proposal
chain = mcmc(θnn, length, lnL, model, nnmodel, nninfo, proposal2, burnin, verbosity)

# get the summary info
acceptance = mean(chain[:,end])
println("acceptance rate: $acceptance")
# compute RMSE
cc = chain[:,1:end-2]
tp = TrueParameters()
chain = Chains(chain[:,1:end-1], names) # convert to Chains type, drop acc. rate
display(chain)
display(plot(chain))
savefig("JDchain.png")
return chain, acceptance
end

