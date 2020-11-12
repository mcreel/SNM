# simple ARMA(1,1) model from Fiorentini et al. 2018, "A spectral EM algorithm for dynamic factor models",
# page 256
using Econometrics, Statistics, Random

function arma11(θ)
    N = 300
    burnin = 50
    α = θ[1]
    β = θ[2]
    σ = sqrt(θ[3])
    data = zeros(N,1)
    xlag = 0.0
    flag = 0.0
    @inbounds for i = 1:(N+burnin)
        f = σ*randn()
        x = α*xlag + f - β*flag
        xlag = x
        flag = f
        if i > burnin
            data[i-burnin] = x
        end    
    end   
    data
end

function auxstat(θ, reps)
    stats = zeros(reps, 13)
    for rep = 1:reps
        x = arma11(θ)
        s = std(x)
        c = cor(x[2:end,:],x[1:end-1])
        n = size(x,1)
        b, varb, u, junk1, rsq  = ols(x[2:end,:], x[1:end-1,:], silent=true)
        ϕ, varϕ, u, junk2, rsq2 = ols(u[2:end,:], u[1:end-1,:], silent=true)
        σ = std(u)
        stats[rep,:] = vcat(s, c, b, varb, rsq, ϕ, varϕ, rsq2, σ, pacf(x,collect(1:4)))
    end
    sqrt(300.0) .* stats
end    

function TrueParameters()
    [0.95, 0.5, 1.0]
end    

function PriorSupport()
    lb = [-0.99, -0.99, 0.0001]
    ub = [0.99, 0.99, 4.0]
    lb,ub
end    

function PriorMean()
    lb, ub = PriorSupport()
    (ub - lb) ./ 2.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    (ub - lb) .* rand(3) + lb
end

# prior just checks that we're in the bounds
function Prior(theta)
    lb, ub = PriorSupport()
    a = 0.0
    if(all((theta .>= lb) .& (theta .<= ub)))
        a = 1.0
    end
    return a
end

