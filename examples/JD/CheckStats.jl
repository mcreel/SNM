using Plots
project="JD"  # set to one of the projects in examples: SV, DPD, ARMA, MN
run_title = "working" # Monte Carlo results written to this file
mcreps = 1000 # how many reps?

# load code
include("JDlib.jl")
include("../../src/SNM.jl")
using BSON:@load

function main()
# load the trained net: note, there are trained nets in the dirs of each project,
# to use those, edit the following line to set the correct path
@load "neural_moments.bson" NNmodel transform_stats_info
trueθ = zeros(mcreps,7)
θhat = zeros(mcreps,7)
for mcrep = 1:mcreps
    θ = PriorDraw()
    trueθ[mcrep,:] = θ
    m = NeuralMoments(θ, auxstat, 1, NNmodel, transform_stats_info) 
    θhat[mcrep,:] = m
end
return trueθ, θhat
end
trueθ, θhat = main()
scatter(trueθ[:,1], θhat[:,1])


