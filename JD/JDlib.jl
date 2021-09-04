using DifferentialEquations, Statistics, Random

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
    TradingDays = burnin+1000 # the sample is 1000 days, also need initial burnin and burnin between samples
    Days = TradingDays + Int(TradingDays/5*2) # add weekends
    MinPerDay = 1440 # minutes per day
    MinPerTic = 10 # minutes between tics, lower for better accuracy
    tics = Int(MinPerDay/MinPerTic) # number of tics in day
    dt = 1/tics # divisions per day
    closing = Int(round(6.5*60/MinPerTic)) # tic at closing
    # solve the diffusion
    μ, κ, α, σ, ρ, λ0, λ1, τ = θ
    u0 = [μ; α]
    prob = Diffusion(μ, κ, α, σ, ρ, u0, (0.0,Days))
    ## jump in log price
    rate(u,p,t) = λ0.*(λ0>0.0) # the prior allows for negative rate, to allow an accumulation at zero
    # jump is random sign times  λ1 times current st. dev.
    affect1!(integrator) = (integrator.u[1] = integrator.u[1].+rand([-1.0,1.0]).*λ1.*exp(integrator.u[2]./2.0))
    jump = ConstantRateJump(rate,affect1!)
    jump_prob = JumpProblem(prob,Direct(), jump)
    # do the simulation
    sol = solve(jump_prob,SRIW1(), dt=dt, adaptive=false, seed=rndseed)
    # get log price, with measurement error
    lnPs = [sol(t)[1] .+ τ.*randn()   for t in dt:dt:Days]
    # get log price at end of trading days. We will compute lag, so loose first
    lnPtrading = zeros(TradingDays+1)
    RV = zeros(TradingDays+1)
    BV = zeros(TradingDays+1)
    DayofWeek = 0 # counter for day of week
    TradingDay = 0 # counter for trading days
    Day = 0
    lnPlag = 0.0
    @inbounds while TradingDay < TradingDays
        Day +=1
        DayofWeek +=1
        # set day of week, and record if it's a trading day
        if DayofWeek<6
            TradingDay +=1 # advance trading day
            # compute realized measures
            t1 = 0.0
            t2 = 0.0
            for tic = 1:closing
                lnP = lnPs[(Day-1)*tics+tic]
                ret = lnP - lnPlag 
                lnPlag = lnP # update lag, this is inter-day for the first
                t2 = t1 # one lag
                t1 = abs(ret) # current
                # RV measures
                if tic > 1
                    RV[TradingDay] += ret*ret
                    BV[TradingDay] += t1*t2
                end
            end    
            lnPtrading[TradingDay] = lnPs[(Day-1)*tics+closing]
        end
        if DayofWeek==7 # restart the week if Sunday
            DayofWeek = 0
        end
    end
    rets = lnPtrading[burnin+1:burnin+1000] .- lnPtrading[burnin:burnin+1000-1] # inter-day returns
    RV = RV[burnin+1:burnin+1000]
    BV = (pi/2.0) .* BV[burnin+1:burnin+1000]
    [rets RV BV]
end

# auxstats, using simulated data
@views function auxstat(θ, reps)
    auxstat.(JDmodel(θ, 50, rand(1:Int64(1e12))) for i = 1:reps)
end

# auxstats, given data
@views function auxstat(data)
    rets = data[:,1]
    # impose in prior that returns should not be too extreme (based on plot of data)
    if std(rets) > 10.0
        return fill(NaN,25)
    else   
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
        jumpsize2 = log.(std(rets[jump]) .+ 1.0) - log.(std(rets[nojump]) .+ 1.0) # limit outliers
        njumps = mean(jump[3:end])
        # ρ
        X = [ones(n-1) BV[2:end] BV[1:end-1]]
        y = rets[2:end]
        βrets = X\y
        ϵrets = y-X*βrets
        σrets = std(ϵrets)
        κrets = std(log.(ϵrets.^2.0))
        # normal volatility: κ, α and σ
        X = [ones(n-2) BV[1:end-2] BV[2:end-1]]
        y = BV[3:end]
        βvol = X\y
        ϵvol = y-X*βvol
        σvol = std(ϵvol)
        κvol = std(log.(ϵvol.^2.0))  # limit outliers
        # jump size
        X = [ones(n) jump BV jump.*BV]
        y = RV
        βjump = X\y
        ϵjump = y-X*βjump
        σjump = std(ϵjump)
        κjump = std(log.(ϵjump.^2.0))
        # jump frequency
        qs = quantile(abs.(rets),[0.5, 0.9])
        qs2 = quantile(RV,[0.5, 0.9])
        qs3 = quantile(BV,[0.5, 0.9])
        return sqrt(n)*vcat(βrets, βvol, βjump,σrets, σvol, σjump,κrets, κvol, κjump, mean(RV) - mean(BV), jumpsize, jumpsize2, qs[2]/qs[1], qs2[2]/qs2[1], qs3[2]./qs3[1], qs2 ./ qs3, njumps)
    end
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
    lb = [-0.1, 0.001, -5.0, 0.01, -0.99, -0.02,  2.0, -0.02]
    ub = [0.1,  0.3, 1.0, 1.5,  -0.3, 0.05, 5.0, 0.05]
    lb,ub
end    

# check if parameter is in support.
function InSupport(θ)
    lb,ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub)
end

# prior checks that we're in the bounds, and that the unconditional std. dev. of log vol is not too high
# returns 1 if this is true, zero otherwise. Value is not important, as it's constant
function Prior(θ)
    InSupport(θ) ? 1.0 : 0.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub + lb) ./ 2.0
end


