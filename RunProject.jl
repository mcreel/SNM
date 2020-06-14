using Pkg
Pkg.activate("../")

run_title = "baseline"
mcreps = 500

# this is the code for the DFM model
include("ARMA/ARMAlib.jl")
global const θtrue = [0.95, 0.5, 1.0]

# this is the code for the DPD model
#include("DPD/DPDlib.jl")
#global const θtrue = [0.6, 1.0, 2.0]

# this is the code for the SV  model
#include("SV/SVlib.jl")
#global const θtrue = [exp(-0.736/2.0), 0.9, 0.363]


include("lib/MakeData.jl")
include("lib/Transform.jl")
include("lib/Train.jl")
include("lib/Analyze.jl")
include("lib/MCMC.jl")

function RunProject()
lb, ub = PriorSupport()
nParams = size(lb,1)
TrainingTestingSize = Int64(nParams*2*1e4) # 25,000 training and testing for each param

# generate the raw training data
MakeData(TrainingTestingSize)
# transform the raw statistics, and split out params and stats
info = Transform()
writedlm("info", info)
# when this is done, can delete raw_data.bson
# train the net using the transformed training/testing data
Train(TrainingTestingSize)
# when this is done, can delete cooked_data.bson
results_GMM = zeros(mcreps,4*nParams)
results_raw = zeros(mcreps,4*nParams)
results_NN = zeros(mcreps,4*nParams)
results_no_Jacobian = zeros(mcreps,4*nParams)
info = readdlm("info")
for mcrep = 1:mcreps
    # generate a draw at true params
    m = ILSNM_model(θtrue)    
    # do full statistic Bayesian GMM estimation
    @time chain, θmile = MCMC(m, false, info, false)
    results_GMM[mcrep,:] = vcat(θmile, Analyze(chain))
     # do full statistic MSM Bayesian estimation
    @time chain, θmile = MCMC(m, false, info, true)
    results_raw[mcrep,:] = vcat(θmile, Analyze(chain))
    # do NN statistic MSM Bayesian estimation
    @time chain, θmile = MCMC(m, true, info, true)
    results_NN[mcrep,:] = vcat(θmile, Analyze(chain))
    # do NN statistic no Jacobian MSM Bayesian estimation
    @time chain, θmile = MCMC(m, true, info, false)
    results_no_Jacobian[mcrep,:] = vcat(θmile, Analyze(chain))
    println("__________ replication: ", mcrep, "_______________")
    println("Results so far, GMM, raw stat")
    dstats(results_GMM[1:mcrep,:])
    println("Results so far, raw stat")
    dstats(results_raw[1:mcrep,:])
    println("Results so far, NN stat")
    dstats(results_NN[1:mcrep,:])
    println("Results so far, NN stat, no Jacobian")
    dstats(results_no_Jacobian[1:mcrep,:])
    println("____________________________")
end
writedlm(run_title*"_GMM", results_GMM)
writedlm(run_title*"_raw", results_raw)
writedlm(run_title*"_NN", results_NN)
writedlm(run_title*"_no_Jacobian", results_no_Jacobian)
end
RunProject()
