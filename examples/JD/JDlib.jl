using DifferentialEquations, Statistics, Econometrics

function Diffusion(μ0,μ1,κ,α,σ,ρ,u0,tspan)
    f = function (du,u,p,t)
        du[1] = μ0 + μ1*(u[2]-α)/σ # drift in log prices
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
    μ0 = 0.0
    μ1 = 0.0
    κ = 0.3
    α = 0.3
    σ = 0.7
    ρ = -0.7
    λ0 = 0.005 # jump rate per day
    λ1 = 4.0 # scaling factor st. dev. of jumps
    return [μ0, μ1, κ, α, σ, ρ, λ0, λ1]
end


function PriorSupport()
    lb = [-0.1, -0.1, 0.0, -1.0, 0.01, -0.999, 0.0,  3.0]
    ub = [0.1,   0.1,  0.5, 3.0, 3.0,  0.0, 0.05, 5.0]
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
μ0, μ1, κ, α, σ, ρ, λ0, λ1 = θ
u0 = [0.0;α]
prob = Diffusion(μ0, μ1, κ, α, σ, ρ, u0, (0.0,Days))
## jump in log price
rate(u,p,t) = λ0
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
    rets, RV, MedRV, ret0, Monday = dgp(θ,reps)
    RV = log.(RV)
    MedRV = log.(MedRV)
    n = Int(size(rets,1)/reps)
    stats = zeros(reps,13)
    #@inbounds Threads.@threads for rep = 1:reps
    @inbounds for rep = 1:reps
        included = n*rep-n+1:n*rep
        βret0,junk,junk  = lsfit(abs.(ret0[included]), [ones(n)  Monday[included]]) # filter out weekend effect
        # drift: μ0 and μ1, also ρ
        X = [ones(n,1) rets[included] MedRV[included]][1:end-1,:]
        y = rets[included][2:end]
        βrets = X\y
        ϵrets = y-X*βrets
        σrets = std(ϵrets)
        # volatility
        y = MedRV[included][2:end]
        βvol = X\y
        ϵvol = y-X*βvol
        σvol = std(ϵvol)
        # leverage
        leverage = cor(ϵvol, ϵrets)
        # dstats
        stats[rep,:] = vcat(βret0, βrets, βvol, σrets, σvol, leverage, mean(RV[included]), mean(MedRV[included]))'
    end
    return stats
end


