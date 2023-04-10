using SimulatedNeuralMoments, Flux, MCMCChains, StatsPlots, DelimitedFiles
using Turing, MCMCChains, AdvancedMH 
using BSON:@save
using BSON:@load
include("JDlib.jl")

function EstimateJD(TrainTestSize=1, Epochs=1000)
    lb, ub = PriorSupport()
    # fill in the structure that defines the model
    model = SNMmodel("SP500 estimation", lb, ub, InSupport, PriorDraw, auxstat)

    # train the net, and save it and the transformation info
    transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
    transformed_prior = transformed(@Prior, transf) # the transformed prior
    nnmodel, nninfo = MakeNeuralMoments(model, transf, TrainTestSize=TrainTestSize, Epochs=Epochs)
    @save "neuralmodel.bson" nnmodel nninfo  # use this line to save the trained neural net 
    @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net

    # define the neural moments using the real data
    data = readdlm("sp500.txt")
    m = NeuralMoments(auxstat(data), nnmodel, nninfo)
    # the raw NN parameter estimate
    θhat = invlink(@Prior, m)
    @show θhat
    # setting for sampling
    names = ["μ","κ","α","σ","ρ","λ₀","λ₁","τ"]
    S = 100
    covreps = 500
    length = 5000
    nchains = 2
    burnin = 100
    tuning = 0.4
    # the covariance of the proposal (scaled by tuning)
    junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo)
@show Σp
    @model function MSM(m, S, model)
        θt ~ transformed_prior
        if !InSupport(invlink(@Prior, θt))
            Turing.@addlogprob! -Inf
            return
        end
        # sample from the model, at the trial parameter value, and compute statistics
        mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
        m ~ MvNormal(mbar, Symmetric(Σ))
    end

    chain = sample(MSM(m, S, model),
        MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(m,1)), tuning*Σp))),
        MCMCThreads(), length, nchains; init_params=Iterators.repeated(m), discard_initial=burnin)

    # transform back to original domain
    chain = Array(chain)
    acceptance = size(unique(chain[:,1]),1)[1] / size(chain,1)
    println("acceptance rate: $acceptance")
    for i = 1:size(chain,1)
        chain[i,:] = invlink(@Prior, chain[i,:])
    end
    chain = Chains(chain, names)
    display(chain)
    display(plot(chain))
    savefig("chain.png")

    return chain, θhat, Σp
end
chain, θhat, Σp = EstimateJD()
writedlm("chain.txt", Array(chain))
writedlm("θhat.txt", θhat)
writedlm("Σp", Σp)


