# this function transforms the input statistics
# to limit outliers, using quantiles

using Econometrics, StatsBase, DelimitedFiles
using BSON: @load
using BSON: @save

function transform(data, info)
    q01 = info[:,1]
    q50 = info[:,2]
    q99 = info[:,3]
    iqr = info[:,4]
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

function Transform()
    # get number of parameters
    lb,ub = PriorSupport()
    nParams = size(lb,1)
    @load "raw_data.bson" data
    params = data[:,1:nParams]
    statistics = data[:,nParams+1:end]
    info = create_transformation(statistics)
    statistics = transform(statistics, info)
    @save "cooked_data.bson" params statistics
    return info
end    
