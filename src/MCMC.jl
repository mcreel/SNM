# This does extremum GMM and then MCMC using the NN estimate as the statistic
using Flux, Econometrics, LinearAlgebra, Statistics, DelimitedFiles
# MVN random walk, or occasional draw from prior
function proposal(current, cholV)
    current + cholV*randn(size(current))
end

function MCMC(θnn, auxstat, NNmodel, info; verbosity = false, nthreads=1, rt=0.5)
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    reps = 10 # replications at each trial parameter
    covreps = 500 # replications used to compute weight matrix
    # use a rapid SAMIN to get good initialization values for chain
    obj = θ -> -1.0*H(θ, θnn, 10, auxstat, NNmodel, info)
    if verbosity == true
        sa_verbosity = 2
    else
        sa_verbosity = 0
    end    
    θsa, junk, junk, junk = samin(obj, θnn, lb, ub; coverage_ok=0, maxevals=1000, verbosity = sa_verbosity, rt = rt)
    # get covariance estimate
    Σ = EstimateΣ(θsa, covreps, auxstat, NNmodel, info) 
    Σinv = inv((1.0+1/reps).*Σ)
    # define things for MCMC
    lnL = θ -> H(θ, θnn, reps, auxstat, NNmodel, info, Σinv)
    ChainLength = Int(1000/nthreads)
    # set up the initial proposal
    try
        P = ((cholesky(Σ)).U)' # transpose it here 
    catch
        P = diagm(diag(Σ))
    end
    Proposal = θ -> proposal(θ, P)
    # initial short chain to tune proposal
    chain = mcmc(θsa, ChainLength, 0, Prior, lnL, Proposal, verbosity, nthreads)
    # loops to tune proposal
    Σ = NeweyWest(chain[:,1:nParams])
    tuning = 1.0
    MC_loops = 5
    @inbounds for j = 1:MC_loops
        P = try
            P = (cholesky(Σ)).U
        catch
            P = diagm(diag(Σ))
        end    
        Proposal = θ -> proposal(θ,tuning*P) # random walk MVN proposal
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
    return chain[:,1:nParams], θnn
end
