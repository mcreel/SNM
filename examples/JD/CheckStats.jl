project="JD"  # set to one of the projects in examples: SV, DPD, ARMA, MN
run_title = "working" # Monte Carlo results written to this file
mcreps = 1000 # how many reps?

# load code
include("examples/"*project*"/"*project*"lib.jl")
include("src/SNM.jl")
using BSON:@load

function main()
# load the trained net: note, there are trained nets in the dirs of each project,
# to use those, edit the following line to set the correct path
@load "neural_moments.bson" NNmodel transform_stats_info
for mcrep = 1:mcreps
    θ = PriorDraw()
    m = NeuralMoments(θ, auxstat, 1, NNmodel, transform_stats_info)    
    prettyprint([θ m],["true" "estimated"],"")
end
end
main()


