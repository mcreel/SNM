using Flux, Statistics

# bounds by quantiles, and standardizes and normalizes around median
function TransformStats(data, info)
    q01,q50,q99,iqr = info
    data = max.(data, q01')
    data = min.(data, q99')
    data = (data .- q50') ./ iqr'
    return data
end

# a draw of neural moments
function NeuralMoments(θ, NNmodel, info)
    Float64.(NNmodel(TransformStats(auxstat(θ)', info)'))
end        

# estimate covariance
function EstimateΣ(θ, S, NNmodel, info)
    ms = zeros(S, size(θ,1))
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = NeuralMoments(θ, NNmodel, info)
    end
    Σ = cov(ms)
end

# method with identity weight
function H(θ, m, S, NNmodel, info)
    k = size(θ,1)
    invΣ = Matrix(1.0I, k, k)
    H(θ, m, S, NNmodel, info, invΣ)
end    

# log likelihood (GMM-form) with fixed weight matrix
function H(θ, m, S, NNmodel, info, invΣ)
    mbar = zeros(size(m))
    Threads.@threads for s = 1:S
        mbar .+= NeuralMoments(θ, NNmodel, info)
    end
    x = m - mbar/S
    -0.5*dot(x,invΣ*x)
end
