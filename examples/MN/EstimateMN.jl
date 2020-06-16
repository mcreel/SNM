include("../../src/SNM.jl")
include("../../src/MCMC.jl") # the specialized MCMC using net 
include("MNlib.jl")
using BSON:@load

function EstimateMN()
    @load "neural_moments.bson" NNmodel transform_stats_info
    m = NeuralMoments(auxstat(TrueParameters()), NNmodel, transform_stats_info)    
    @time chain, θhat = MCMC(m, NNmodel, transform_stats_info)
    p1 = npdensity(chain[:,1])
    plot!(label="")
    p2 = npdensity(chain[:,2])
    plot!(label="")
    p3 = npdensity(chain[:,3])
    plot!(label="")
    p4 = npdensity(chain[:,4])
    plot!(label="")
    p5 = npdensity(chain[:,5])
    plot!(label="")
    plot(p1,p2,p3,p4,p5, layout=(5,1))
    plot!(label="")
end
EstimateMN()


