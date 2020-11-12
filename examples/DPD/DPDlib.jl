using Econometrics, Statistics, Random

# stacks up the data for each agent, result is N blocks of T
function dgp(θ)
    N = 100
    T = 5
    data = zeros(N*T,3)
    datadm = zeros(N*T,3)
    for i = 1:N
        @inbounds data[i*T-T+1:i*T,:], datadm[i*T-T+1:i*T,:] = dgp_agent(θ)
    end
    return data, datadm
end

# DGP for individual agent
function dgp_agent(θ)
    ρ = θ[1]
    β = θ[2]
    σ = sqrt(θ[3])
    T = 5
    data = zeros(T,3)
    # individual effect
    α = randn()
    y = α + σ/sqrt(1.0 - ρ*ρ)*randn()
    for t = 1:T
        data[t,3] = y # lagged y
        x = randn()
        y = α + ρ*y + β*x + σ*randn()
        data[t,1] = y
        data[t,2] = x
    end
    datadm = data .- mean(data,dims=1) # demean here, so as not to have to construct the big matrix
    return data, datadm
end

function auxstat(θ, reps)
    ρ, β, σ = θ
    stats = zeros(reps, 7)
    T = 5
    for rep = 1:reps
        data, datadm = dgp(θ)
        # demeaned data
        y = datadm[:,1]
        x = datadm[:,2]
        ylag = datadm[:,3]
        b1, junk, u = lsfit(y, [x ylag])
        s1 = sqrt(mean(u.*u))
        y = data[:,1]
        x = data[:,2]
        ylag = data[:,3]
        n = size(y,1)
        b2, junk, u = lsfit(y, [ones(n) x ylag])
        s2 = sqrt(mean(u.*u))
        stats[rep, :] = 10.0*vcat(b1, s1, b2, s2)
    end
    return stats
end    

function TrueParameters()
    [0.6, 1.0, 2.0]
end    

function PriorSupport()
    lb = [0.0, -5.0, 0.01]
    ub = [0.999, 5.0, 10.0]
    lb,ub
end    

function PriorMean()
    lb, ub = PriorSupport()
    (ub - lb) ./ 2.0
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

function PriorDraw()
    lb, ub = PriorSupport()
    (ub-lb).*rand(size(lb,1)) + lb
end    


