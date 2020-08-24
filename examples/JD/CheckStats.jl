# check for 45º true-fitted plots for all parameters
# to verify that stats give good identification
using Pkg
Pkg.activate("../../")
using Plots, Econometrics
mcreps = 500 # how many reps?

# load code
include("JDlib.jl")
include("../../src/SNM.jl")
using BSON:@load

function main()
@load "neural_moments.bson" NNmodel transform_stats_info
trueθ = zeros(mcreps,8)
θhat = zeros(mcreps,8)
Threads.@threads for mcrep = 1:mcreps
    println("mc rep: ", mcrep)
    θ = PriorDraw()
    trueθ[mcrep,:] = θ
    θhat[mcrep,:] = NeuralMoments(θ, auxstat, 1, NNmodel, transform_stats_info)
end    
return trueθ, θhat
end
trueθ, θhat = main()
savefig(scatter(trueθ[:,1], θhat[:,1]),"mu.svg")
savefig(scatter(trueθ[:,2], θhat[:,2]),"kappa.svg")
savefig(scatter(trueθ[:,3], θhat[:,3]),"alpha.svg")
savefig(scatter(trueθ[:,4], θhat[:,4]),"sigma.svg")
savefig(scatter(trueθ[:,5], θhat[:,5]),"rho.svg")
savefig(scatter(trueθ[:,6], θhat[:,6]),"lambda0.svg")
savefig(scatter(trueθ[:,7], θhat[:,7]),"lambda1.svg")
savefig(scatter(trueθ[:,8], θhat[:,8]),"tau.svg")
for i = 1:8
    ols(θhat[:,i],[ones(mcreps) trueθ[:,i]])
end
