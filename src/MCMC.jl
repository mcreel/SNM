# This does MLE and then MCMC, either using raw statistic, or using NN transform,
# depending on the argument usenn
using Flux, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
include("SNM.jl")

# uniform random walk in one dimension
function proposal1(current, tuning)
    trial = copy(current)
    i = rand(1:size(trial,1))
    trial[i] += tuning[i].*randn()
    return trial
end

# MVN random walk, or occasional draw from prior
function proposal2(current, cholV)
    current + cholV'*randn(size(current))
end

function MCMC(m, auxstat, NNmodel, info; verbosity = false, nthreads=1, rt=0.5)
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    # use a rapid SAMIN to get good initialization values for chain
    obj = θ -> -1.0*H(θ, m, 10, auxstat, NNmodel, info)
    if verbosity == true
        sa_verbosity = 2
    else
        sa_verbosity = 0
    end    
    θhat, junk, junk, junk = samin(obj, m, lb, ub; coverage_ok=0, maxevals=100000, verbosity = sa_verbosity, rt = rt)
    # get covariance estimate
    reps = 10
    Σinv = inv((1.0+1/reps).*EstimateΣ(θhat, 100, auxstat, NNmodel, info))
    # define things for MCMC
    lnL = θ -> H(θ, m, reps, auxstat, NNmodel, info, Σinv)
    ChainLength = Int(1000/nthreads)
    MCMCburnin = 0
    tuning = 0.2/sqrt(12.0)*(ub-lb) # two tenths of a standard. dev. of prior
    Proposal = θ -> proposal1(θ, tuning)
    chain = mcmc(θhat, ChainLength, MCMCburnin, Prior, lnL, Proposal, verbosity, nthreads)
    # now use a MVN random walk proposal with updates of covariance and longer chain
    # on final loop
    Σ = NeweyWest(chain[:,1:nParams])
    tuning = 1.0
    MC_loops = 5
    @inbounds for j = 1:MC_loops
        P = try
            P = (cholesky(Σ)).U
        catch
            P = diagm(diag(Σ))
        end    
        Proposal = θ -> proposal2(θ,tuning*P)
        if j == MC_loops
            ChainLength = Int(10000/nthreads)
        end    
        θinit = mean(chain[:,1:nParams],dims=1)[:]
        chain = mcmc(θinit, ChainLength, 0, Prior, lnL, Proposal, verbosity, nthreads)
        if j < MC_loops
            accept = mean(chain[:,end])
            if accept > 0.35
                tuning *= 1.5
            elseif accept < 0.25
                tuning *= 0.25
            end
            Σ = 0.5*Σ + 0.5*NeweyWest(chain[:,1:nParams])
        end    
    end
    chain = chain[:,1:nParams]
    return chain, θhat
end
