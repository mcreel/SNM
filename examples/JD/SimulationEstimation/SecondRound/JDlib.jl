using DifferentialEquations, Statistics, Econometrics

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

function TrueParameters()
    μ = 0.02
    κ = 0.2
    α = 0.3
    σ = 0.7
    ρ = -0.7
    λ0 = 0.005 # jump rate per day
    λ1 = 4.0 # scaling factor st. dev. of jumps
    return [μ, κ, α, σ, ρ, λ0, λ1]
end


function PriorSupport()
    lb = [-0.1, 0.001, -1.0, 0.01, -0.99, -0.02,  3.0]
    ub = [0.1,  0.5, 1.0, 2.0,  0.0, 0.05, 6.0]
    lb,ub
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub + lb) ./ 2.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

# prior checks that we're in the bounds, and that the unconditional std. dev. of log vol is not too high
# returns 1 if this is true, zero otherwise. Value is not important, as it's constant
function Prior(θ)
    lb,ub = PriorSupport()
    a = 0.0
    if (all(θ .>= lb) & all(θ .<= ub))
        a = 1.0
    end
    return a
end

function dgp(θ,reps)
TradingDays = 1000*reps    
Days = TradingDays+Int(TradingDays/5*2)+1 # add weekends, plus a day for lag
MinPerDay = 1440 # minutes per day
MinPerTic = 5 # minutes between tics, lower for better accuracy
tics = Int(MinPerDay/MinPerTic) # number of tics in day
dt = 1/tics # divisions per day
closing = Int(round(6.5*60/MinPerTic)) # tic at closing
# parameters
μ, κ, α, σ, ρ, λ0, λ1 = θ
u0 = [0.0;α]
prob = Diffusion(μ, κ, α, σ, ρ, u0, (0.0,Days))
## jump in log price
rate(u,p,t) = λ0.*(λ0>0.0) # the prior allows for negative rate, to allow an accumulation at zero
# jump is random sign times  λ1 times current st. dev.
affect1!(integrator) = (integrator.u[1] = integrator.u[1].+rand([-1.0,1.0]).*λ1.*exp(integrator.u[2]./2.0))
jump = ConstantRateJump(rate,affect1!)
jump_prob = JumpProblem(prob,Direct(), jump)
sol = solve(jump_prob,SRIW1(), dt=dt, adaptive=false)
# find when jumps occur
#jump = sol.t[2:end].==sol.t[1:end-1] # find times where jumps occur
#jump[1] = 0 # this is always true, for some reason, set it false
#jump = vcat(false,jump) 
#jumptimes = sol.t[jump] .* TradingDays ./ Days 
lnPs = [sol(t)[1] for t in dt:dt:Days]
#vol = [sol(t)[2] for t in dt:dt:Days]
# get log price at end of trading days. We will compute lag, so loose first
lnPtrading = zeros(TradingDays+1)
#Volatility = zeros(TradingDays+1) # real latent volatility
RV = zeros(TradingDays+1)
MedRV = zeros(TradingDays+1)
ret0 = zeros(TradingDays+1) # returns at open
Monday = zeros(TradingDays+1)
DayofWeek = 0 # counter for day of week
TradingDay = 0 # counter for trading days
Day = 0
lnPlag = 0.0
@inbounds while TradingDay < TradingDays+1
    Day +=1
    DayofWeek +=1
    # set day of week, and record if it's a trading day
    if DayofWeek<6
        TradingDay +=1 # advance trading day
        # compute realized measures
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        for tic = 1:closing
            lnP = lnPs[(Day-1)*tics+tic]
            ret = lnP - lnPlag 
            lnPlag = lnP # update lag, this is inter-day for the first
            t3 = t2 # two lags
            t2 = t1 # one lag
            t1 = abs(ret) # current
            # compute interday initial return
            if tic == 1
                ret0[TradingDay] = ret
            end
            # RV measures
            if tic > 2
                RV[TradingDay]+=(ret^2.0)
                MedRV[TradingDay] += median([t1,t2,t3])^2.0
            end
        end    
        lnPtrading[TradingDay] = lnPs[(Day-1)*tics+closing]
        #Volatility[TradingDay] = exp(0.5*vol[(Day-1)*tics+closing]) # true volatility
        Monday[TradingDay] = (DayofWeek == 1)
    end
    if DayofWeek==7 # restart the week if Sunday
        DayofWeek = 0
    end
end
rets = lnPtrading[2:end]-lnPtrading[1:end-1] # inter-day returns
RV = RV[2:end]
MedRV = pi/(6.0-4.0*sqrt(3.0) + pi).*MedRV
MedRV = MedRV[2:end]
#Volatility = Volatility[2:end]
Monday = Monday[2:end]
ret0 = ret0[2:end]
#return rets, Volatility, jumptimes, RV, MedRV, ret0, Monday
return rets, RV, MedRV, ret0, Monday
end

# returns reps replications of the statistics
function auxstat(θ, reps)
    not_ok = true
    stats = 0.0
    while not_ok
        rets, RV, MedRV, ret0, Monday = dgp(θ,reps)
        RV = log.(RV)
        MedRV = log.(MedRV)
        jump = RV .> (2.0 .* MedRV)
        nojump = jump .== false
        jump[1:2] .= true
        nojump[1:2] .= true

        n = size(rets,1)
        jumpsize = mean(RV[jump]) - mean(MedRV[jump])
        jumpsize2 = std(rets[jump]) - std(rets[nojump])
        njumps = mean(jump[3:end])

        # look at opening returns, for overnight/weekend jumps
        X = [ones(n-1) MedRV[1:end-1] Monday[2:end]]
        y = abs.(ret0[2:end])
        βret0 = X\y
        u = y - X*βret0
        σ0 = std(u) # larger variance means more frequent jumps
        κ0 = std(u.^2.0)
        # ρ
        X = [ones(n-1) MedRV[2:end] MedRV[1:end-1]]
        y = rets[2:end]
        βrets = X\y
        ϵrets = y-X*βrets
        σrets = std(ϵrets)
        κrets = std(ϵrets.^2.0)
        # normal volatility: κ, α and σ
        X = [ones(n-2) MedRV[1:end-2] MedRV[2:end-1]]
        y = MedRV[3:end]
        βvol = X\y
        ϵvol = y-X*βvol
        σvol = std(ϵvol)
        κvol = std(ϵvol.^2.0)
        # jump size
        X = [ones(n) jump MedRV jump.*MedRV]
        y = RV
        βjump = X\y
        ϵjump = y-X*βjump
        σjump = std(ϵjump)
        κjump = std(ϵjump.^2.0)
        # jump frequency
        qs = quantile(abs.(rets),[0.5, 0.9])
        qs2 = quantile(abs.(ret0),[0.5, 0.9])
        qs3 = quantile(RV,[0.5, 0.9])
        # leverage
        leverage1 = cor(MedRV, rets)
        leverage2 = cor(RV, rets)
        stats = vcat(βret0, βrets, βvol, βjump, σ0, σrets, σvol, σjump, κ0, κrets, κvol, κjump, leverage1, leverage2, mean(RV) - mean(MedRV), jumpsize, jumpsize2, qs[2]/qs[1], qs2[2]/qs2[1], qs3[2]/qs3[1],  njumps, mean(rets), mean(ret0))'
        # needs updating!
        # bret0 1:3
        # brets 4:5
        # bvol 6:8
        # bjump 9:12
        # sig0 13
        # sigrets 14
        # sigvol 15
        # sigjump 16
        # k0 17
        # krets 18
        # kvol 19
        # kjump 20
        # l1 21
        # l2 22
        # mean RV - mean MedRV 23
        # jumpsize 24
        # jumpsize2 25
        # qs1 26
        # qs2 27 
        # qs3 28
        # njumps 29 
        # mean rets 30
        # mean ret0 31
        #
        not_ok = any(isnan.(stats))
    end
    return stats
end
