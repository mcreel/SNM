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
function NeuralMoments(θ, auxstat, reps, NNmodel, info)
    z = auxstat(θ, reps)
    Float64.(NNmodel(TransformStats(z, info)'))
end        

# estimate covariance
function EstimateΣ(θ, reps, auxstat, NNmodel, info)
    ms = NeuralMoments(θ, auxstat, reps, NNmodel, info)
    Σ = cov(ms')
end

# method with identity weight
function H(θ, m, reps, auxstat, NNmodel, info)
    k = size(θ,1)
    invΣ = Matrix(1.0I, k, k)
    H(θ, m, reps, auxstat, NNmodel, info, invΣ)
end    

# log likelihood (GMM-form) with fixed weight matrix
function H(θ, m, reps, auxstat, NNmodel, info, invΣ)
    x = m - vec(mean(NeuralMoments(θ, auxstat, reps, NNmodel, info), dims=2))
    -0.5*dot(x,invΣ*x)
end
