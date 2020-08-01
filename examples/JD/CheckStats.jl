using Plots
mcreps = 10 # how many reps?

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
    println("mc rep: ", mcrep)
    θ = PriorDraw()
    trueθ[mcrep,:] = θ
    θhat[mcrep,:] = mean(NeuralMoments(θ, auxstat, 10, NNmodel, transform_stats_info), dims=2)
end    
return trueθ, θhat
end

trueθ, θhat = main()
scatter(trueθ[:,2], θhat[:,2])


