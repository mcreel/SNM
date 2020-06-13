#using Pkg
#Pkg.activate("../")

run_title = "working"
mcreps = 1000

# this code only estimates the selected version,
# which is by default with neural net and without Jacobian
# the code in v1.0 estimates all 4 versions
# 
# also, this uses the extemum estimator
# to compute an estimate of the covariance
# of the moments, which is fixed over the 
# MCMC iterations, like a 2 step GMM esimator,
# rather than a CUE estimator. This is faster.

# this is the code for the DFM model
#include("ARMA/ARMAlib.jl")
#global const θtrue = [0.95, 0.5, 1.0]

# this is the code for the DPD model
#include("DPD/DPDlib.jl")
#global const θtrue = [0.6, 1.0, 2.0]

# this is the code for the SV  model
include("SV/SVlib.jl")
global const θtrue = [exp(-0.736/2.0), 0.9, 0.363]

# this is the code for the mixture of normals model
#include("MN/MNlib.jl")
#global const θtrue = [1.0, 0.0, 0.2, 2.0, 0.4]


include("lib/MakeData.jl")
include("lib/Transform.jl")
include("lib/Train.jl")
include("lib/Analyze.jl")
include("lib/MCMC.jl")

function RunProject()
lb, ub = PriorSupport()
nParams = size(lb,1)
TrainingTestingSize = Int64(nParams*2*1e4) # 20,000 training and testing for each param

# generate the raw training data
#MakeData(TrainingTestingSize)
# transform the raw statistics, and split out params and stats
#info = Transform()
#writedlm("info", info)
# when this is done, can delete raw_data.bson
# train the net using the transformed training/testing data
#Train(TrainingTestingSize)
# when this is done, can delete cooked_data.bson
results = zeros(mcreps,4*nParams)
info = readdlm("info")
for mcrep = 1:mcreps
    # generate a draw at true params
    m = ILSNM_model(θtrue)    
    @time chain, θmile = MCMC(m, true, info)
    results[mcrep,:] = vcat(θmile, Analyze(chain))
    println("__________ replication: ", mcrep, "_______________")
    println("Results so far")
    println("parameter estimates")
    dstats(results[1:mcrep,1:nParams]; short=true)
    println("CI coverage")
    clabels = ["99%","95%","90%"]
    prettyprint(reshape(mean(results[1:mcrep,nParams+1:end],dims=1),nParams,3),clabels)
    println("____________________________")
end
writedlm(run_title, results)
end
RunProject()
