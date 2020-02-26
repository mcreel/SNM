# makes the training and testing data

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
            ok = Prior(θ) # note: some draws in bounds may not be in support
            if ok != 1.0
            end    
        end    
        W = ILSNM_model(θ) # draw the raw statistics
        if s == 1
            data = zeros(S, size(vcat(θ, W),1))
        end
        data[s,:] = vcat(θ, W)
    end
    # save needed items with standard format
    @save "raw_data.bson" data
    return nothing
end
