using Econometrics, StatsBase
using BSON: @load
using BSON: @save

function MakeData(S)
    data = 0.0
    datadesign = 0.0
    lb,ub = PriorSupport()
    # training and testing
    for s = 1:S
        ok = 0.0
        θ = lb # initialize
        while ok != 1.0
            v = rand(size(lb,1))
            θ = v.*(ub-lb) + lb
            ok = Prior(θ)
            if ok != 1.0
            end    
        end    
        m = ILSNM_model(θ)
        if s == 1
            data = zeros(S, size(vcat(θ, m),1))
        end
        #data[s,:] = vcat(-log.(1.0./v .- 1.0), m)
        data[s,:] = vcat(θ, m)
    end
    # save needed items with standard format
    @save "raw_data.bson" data
    return nothing
end
