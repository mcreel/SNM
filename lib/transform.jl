# bounds by quantiles, and standardizes and normalizes around median
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

