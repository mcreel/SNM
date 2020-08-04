# load code
include("JDlib.jl")
include("../../src/SNM.jl")
include("../../src/MakeNeuralMoments.jl") # the specialized MCMC using net 

function main()
# generate the trained net: comment out when done for the chosen model
nParams = size(PriorSupport()[1],1)
#TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each parameter
TrainingTestingSize = Int64(2*1e3) # 20,000 training and testing for each parameter
MakeNeuralMoments(auxstat, TrainingTestingSize) # already done for the 4 examples
end
@time main()


