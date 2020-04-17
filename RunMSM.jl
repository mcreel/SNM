using Pkg, DelimitedFiles
Pkg.activate("../")

run_title = "MSM"
mcreps = 500

# this is the code for the DFM model
#include("ARMA/ARMAlib.jl")
#global const θtrue = [0.95, 0.5, 1.0]

# this is the code for the DPD model
#include("DPD/DPDlib.jl")
#global const θtrue = [0.6, 1.0, 2.0]

# this is the code for the SV  model
include("SV/SVlib.jl")
global const θtrue = [exp(-0.736/2.0), 0.9, 0.363]

include("lib/MSM.jl")
include("lib/AnalyzeMSM.jl")

function RunMSM()
lb, ub = PriorSupport()
nParams = size(lb,1)

#results_raw = zeros(mcreps,6*nParams)
results_raw = zeros(mcreps,4*nParams)
for mcrep = 1:mcreps
    # generate a draw at true params
    m = ILSNM_model(θtrue)    
    # do full statistic MSM Bayesian estimation
    @time θmile, se = MSM(m)
    results_raw[mcrep,:] = vcat(θmile,AnalyzeMSM(θmile, se))
    println("__________ replication: ", mcrep, "_______________")
    println("Results so far, raw stat")
    dstats(results_raw[1:mcrep,:])
    println("____________________________")
end
writedlm(run_title*"_results", results_raw)
end
RunMSM()
