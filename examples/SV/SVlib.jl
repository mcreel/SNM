using Econometrics, Statistics, Random

function auxstat(θ, reps)
    stats = zeros(reps,11)
    for rep = 1:reps
        n = 500
        burnin = 100
        y = SVmodel(θ, n, burnin)
        s = std(y)
        y = abs.(y)
        m = mean(y)
        s2 = std(y)
        y = y ./ s2
        k = std((y).^2.0)
        c = cor(y[1:end-1],y[2:end])
        # ratios of quantiles of moving averages to detect clustering
        q = try
            q = quantile((ma(y,3)[3:end]), [0.25, 0.75])
        catch
            q = [1.0, 1.0]
        end
        c1 = log(q[2]/q[1])
        stats[rep,:] = sqrt(Float64(n))*vcat(m, s, s2, k, c, c1, HAR(y))'
    end
    return stats
end


# version which generates shock internally
function SVmodel(θ, n, burnin)
    ϕ = θ[1]
    ρ = θ[2]
    σ = θ[3]
    hlag = 0.0
    ys = zeros(n)
    @inbounds for t = 1:burnin+n
        h = ρ*hlag + σ*randn()
        y = ϕ*exp(h/2.0)*randn()
        if t > burnin 
            ys[t-burnin] = y
        end    
        hlag = h
    end
    ys
end

# auxiliary model: HAR-RV
# Corsi, Fulvio. "A simple approximate long-memory model
# of realized volatility." Journal of Financial Econometrics 7,
# no. 2 (2009): 174-196.
function HAR(y)
    ylags = lags(y,10)
    X = [ones(size(y,1)) ylags[:,1]  mean(ylags[:,1:4],dims=2) mean(ylags[:,1:10],dims=2)]
    # drop missings
    y = y[11:end]
    X = X[11:end,:]
    βhat = X\y
    σhat = std(y-X*βhat)     
    vcat(βhat,σhat)
end

function TrueParameters()
    [exp(-0.736/2.0), 0.9, 0.363]
end

function PriorSupport()
    lb = [0.05, 0.0, 0.05]
    ub = [2.0, 0.999, 1.0]
    lb,ub
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub - lb) ./ 2.0
end

# prior checks that we're in the bounds, and that the unconditional std. dev. of log vol is not too high
# returns 1 if this is true, zero otherwise. Value is not important, as it's constant
function Prior(θ)
    lb,ub = PriorSupport()
    a = 0.0
    if (all(θ .>= lb) & all(θ .<= ub))
        if (θ[3]/sqrt(1.0 - θ[2]^2.0) < 5.0)
            a = 1.0
        end    
    end
    return a
end

function PriorDraw()
    lb, ub = PriorSupport()
    ok = false
    θ = 0.0
    while !ok
        θ = (ub-lb).*rand(size(lb,1)) + lb
        ok = Prior(θ)==1.0
    end
    return θ
end    


