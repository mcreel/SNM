include("TransformStats.jl")

# estimate covariance
function EstimateΣ(θ, m, S, NNmodel, info)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = Float64.(NNmodel(TransformStats(auxstat(θ)', info)'))
    end
    Σ = cov(ms)
end

# method with identity weight
function H(θ, m, S, NNmodel, info)
    invΣ = eye(size(θ,1))
    H(θ, m, S, NNmodel, info, invΣ)
end    

# log likelihood (GMM-form) with fixed weight matrix
function H(θ, m, S, NNmodel, info, invΣ)
    mbar = zeros(size(m))
    Threads.@threads for s = 1:S
        mbar .+= Float64.(NNmodel(TransformStats(auxstat(θ)', info)'))
    end
    x = m - mbar/S
    -0.5*dot(x,invΣ*x)
end

