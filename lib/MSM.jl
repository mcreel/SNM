# This does ordinary gradient-based IL, and gets CIs from asymptotic theory
using Econometrics, LinearAlgebra, Statistics, Calculus
include("MSMlib.jl")

function MSM(m)
    S = 100 # number of simulations
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    θinit = PriorMean() # prior mean as initial θ
    obj = θ -> MSMobj(θ, m, S)
    # use a rapid SAMIN to get good initialization values for chain
    θmile, junk, junk, junk = samin(obj, θinit, lb, ub; coverage_ok=0, maxevals=100000, verbosity = 0, rt = 0.5)

    # compute the estimated standard errors and CIs
    Moments = θ -> MSMmoments(θ, m, S)
    D = (Calculus.jacobian(Moments, vec(θmile), :central))
    Sigma = θ -> MSMsigma(θ, m, S)
    W = inv(Sigma(θmile))
    V = inv(D'*W*D) 
    se = 500.0*sqrt.(diag(V))
    return θmile, se
end
