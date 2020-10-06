# The auction model, inputs are parameters and sample size
function AuctionModel(theta, n)
    # the model: any zero max bids are rejected (so, it's part of prior)
	theta1, theta2 = theta
	N = 6
	ok = false
    b = [0.0, 0.0]
    x = zeros(2,2)
    while !ok
        # quality of good
        x = 4.0*(rand(n)).^2.0
        # valuations drawn from exponential mean phi
        phi = exp.(theta1 .+ theta2*x)
        # highest valuation
        v = -log.(minimum(rand(n,6);dims=2)).*phi
        # get winning bid (this is from CAS solution, which is easy when N in known)
        z = v./phi
        D = exp.(-5.0*z).*(60.0*exp.(5.0*z) + 300.0*phi .* exp.(4.0*z) - 300.0*phi .* exp.(3.0*z)
            + 200.0*phi .* exp.(2.0*z) - 75.0*phi .* exp.(z) + 12.0*phi)/60.0 - 137.0*phi/60.0
        b = v .- D ./ ((1.0 .- exp.(-v./phi)).^(N-1))
        ok = all(b.>0.0)
    end    
    log.(b), [ones(n) x]
end

# Auxiliary statistic.
function auxstat(theta, reps)
    n = 20 # number of auctions
    stats = zeros(reps, 7)
    for i = 1:reps
        y, x = AuctionModel(theta, n)
        bhat = x\y
        sig = std(y - x*bhat)
        stats[i,:] = sqrt(n)*vcat(bhat, sig, mean(y), std(y), skewness(y), kurtosis(y))
    end
    return stats
end

function TrueParameters()
    [0.5, 0.5]
end    

function PriorSupport()
    lb = [0.0, 0.0]
    ub = [5.0, 5.0]
    lb,ub
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub - lb) ./ 2.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

function Prior(θ)
    lb,ub = PriorSupport()
    a = 0.0
    if (all(θ .>= lb) & all(θ .<= ub))
        a = 1.0
    end
    return a
end



