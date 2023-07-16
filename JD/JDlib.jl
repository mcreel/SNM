using DifferentialEquations, Statistics, Random

isweekday(d::Int)::Bool = (d % 7) % 6 != 0

function Diffusion(μ,κ,α,σ,ρ,u0,tspan)
    f = function (du,u,p,t)
        du[1] = μ # drift in log prices
        du[2] = κ.*(α.-u[2]) # mean reversion in shocks
    end
    g = function (du,u,p,t)
        du[1] = exp(u[2]/2.0)
        du[2] = σ
    end
    Γ = [1.0 ρ;ρ 1.0] # Covariance Matrix
    noise = CorrelatedWienerProcess!(Γ,tspan[1],zeros(2),zeros(2))
    sde_f = SDEFunction{true}(f,g)
    SDEProblem(sde_f,g,u0,tspan,noise=noise)
end

@views function JDmodel(θ, burnin, rndseed=1234)
    Random.seed!(rndseed)
    trading_days = 1000
    days = round(Int, 1.4 * (trading_days + burnin)) # Add weekends (x + x/5*2 = 1.4x)
    min_per_day = 1_440 # Minutes per day
    min_per_tic = 10 # Minutes between tics, lower for better accuracy
    tics = round(Int, min_per_day / min_per_tic) # Number of tics per day
    dt = 1/tics # Divisions per day
    closing = round(Int, 390 / min_per_tic) # Tic at closing (390 = 6.5 * 60)

    # Solve the diffusion
    μ, κ, α, σ, ρ, λ₀, λ₁, τ = θ
    τ = τ*(τ>0)  
    u₀ = [μ; α]
    prob = Diffusion(μ, κ, α, σ, ρ, u₀, (0., days))
    λ₀⁺ = max(0, λ₀) # The prior allows for negative rate, to allow an accumulation at zero

    # # Jump in log price
    rate(u, p, t) = λ₀⁺


    # Jump is random sign time λ₁ times current std. dev.
    function affect!(integrator)
        integrator.u[1] = integrator.u[1] + rand([-1., 1.]) * λ₁ * exp(integrator.u[2] / 2)
        nothing
    end

    jump = ConstantRateJump(rate, affect!)
    jump_prob = JumpProblem(prob, Direct(), jump)

    # Do the simulation
    sol = solve(jump_prob, SRIW1(), dt=dt, adaptive=false)

    # Get log price, with measurement error 
    # Trick: we only need very few log prices, 39 per trading day, use smart filtering
    lnPs = (
        [sol(t)[1] + τ * randn() for t ∈ Iterators.take(p, closing)]
        for (_, p) ∈ Iterators.drop(
            Iterators.filter(
                x -> isweekday(x[1]), 
                enumerate(Iterators.partition(dt:dt:days, tics))), 
            burnin - 1)
    )

    # Get log price at end of trading days We will compute lag, so lose first
    lnP_trading = zeros(Float64, trading_days + 1)
    rv = zeros(Float64, trading_days + 1)
    bv = zeros(Float64, trading_days + 1) 

    p₋₁ = 0.
    @inbounds for (t, p) ∈ enumerate(lnPs)
        r = abs.(diff([p₋₁; p]))
        bv[t] = dot(r[2:end], r[1:end-1])
        rv[t] = dot(r[2:end], r[2:end])
        p₋₁ = p[end]
        lnP_trading[t] = p[end]
    end
    
    [diff(lnP_trading) rv[2:end] π/2 .* bv[2:end]]
end

# auxstats, using simulated data
@views function auxstat(θ, reps)
    auxstat.(JDmodel(θ, 50, rand(1:Int64(1e12))) for i = 1:reps)
end

# auxstats, given data
@views function auxstat(data)
    rets = data[:,1]
    RV = log.(data[:,2])
    BV = log.(data[:,3])
    jump = RV .> (1.5 .* BV)
    nojump = jump .== false
    n = size(data,1)
    # ensure variation
    jump[1:2] .= true
    nojump[1:2] .= true
    # jump stats 
    jumpsize = mean(RV[jump]) - mean(BV[jump])
    jumpsize2 = std(rets[jump]) - std(rets[nojump])
    njumps = mean(jump[3:end])
    # ρ
    X = [ones(n-1) BV[2:end] BV[1:end-1]]
    y = rets[2:end]
    βrets = X\y
    ϵrets = y-X*βrets
    σrets = std(ϵrets)
    κrets = std(abs.(ϵrets))
    # normal volatility: κ, α and σ
    X = [ones(n-2) BV[1:end-2] BV[2:end-1]]
    y = BV[3:end]
    βvol = X\y
    ϵvol = y-X*βvol
    σvol = std(ϵvol)
    κvol = std(abs.(ϵvol))
    # jump size
    X = [ones(n) jump BV jump.*BV]
    y = RV
    βjump = X\y
    ϵjump = y-X*βjump
    σjump = std(ϵjump)
    κjump = std(abs.(ϵjump))
    # jump frequency
    qs = quantile(abs.(rets),[0.5, 0.9])
    qs2 = quantile(RV,[0.5, 0.9])
    qs3 = quantile(BV,[0.5, 0.9])
    return sqrt(n)*vcat(βrets, βvol, βjump,σrets, σvol, σjump,κrets, κvol, κjump, mean(RV) - mean(BV), jumpsize, jumpsize2, qs[2]/qs[1], qs2[2]/qs2[1], qs3[2]./qs3[1], qs2 ./ qs3, njumps)
    # brets 1:3
    # bvol 4:6
    # bjump 7:10
    # sigrets 11
    # sigvol 12
    # sigjump 13
    # krets 14
    # kvol 15
    # kjump 16
    # mean RV - mean BV 17
    # jumpsize 18
    # jumpsize2 19
    # qs1 20
    # qs2 21 
    # qs3 22
    # qs2/qs3 23-24 
    # njumps 25
end

function TrueParameters()
    μ = 0.02
    κ = 0.2
    α = 0.3
    σ = 0.7
    ρ = -0.7
    λ0 = 0.005 # jump rate per day
    λ1 = 4.0 # scaling factor st. dev. of jumps
    τ = 0.005
    return [μ, κ, α, σ, ρ, λ0, λ1, τ]
end

function PriorSupport()
    lb = [-.05, .02, -10.0,  0.1, -.99,  -0.02,  3., -.02] 
    ub = [ .05, .30,   0.,   4.0, -.50,  .05,    6.,  .05]
    lb,ub
end    

# check if parameter is in support.
function InSupport(θ)
    lb,ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub)
end


function Prior(θ)
    InSupport(θ) ? 1. : 0.
end    

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub + lb) ./ 2.0
end


