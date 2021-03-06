using DifferentialEquations, Plots, Statistics

function Diffusion(μ0,μ1,κ,α,σ,ρ,u0,tspan)
    f = function (du,u,p,t)
        du[1] = μ0 + μ1*(u[2]-α)/σ # drift in log prices
        du[2] = κ*(α-u[2]) # mean reversion in shocks
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


function dgp()
# trading days, closing, etc

# assume trading period is 1/3 of day (8 hours) 
# but that latent price evolves continuously
# observed return is daily difference of log price at
# closing time
TradingDays = 1000 # total days in sample
Days = Int(ceil(TradingDays*7/5))+3  # calendar days
MinPerDay = 1440 # minutes per day
MinPerTic = 5 # minutes between tics, lower for better accuracy
tics = Int(MinPerDay/MinPerTic) # number of tics in day
dt = 1/tics # divisions per day
closing = Int(floor(tics/3)) # closing tic: closing happens after 1/3 of day
# parameters
μ0 = 0.0
μ1 = 0.0
κ = 0.1
α = 0.15
σ = 0.15
ρ = -0.7
λ0 = 0.005 # jump rate per day
λ1 = 4.0 # scaling factor st. dev. of jumps
u0 = [0.0;α]
prob = Diffusion(μ0, μ1, κ, α, σ, ρ, u0, (0.0,Days))

## jump in price
rate(u,p,t) = λ0
# jump is normal with st. dev. equal to λ1 times current st. dev.
affect1!(integrator) = (integrator.u[1] = integrator.u[1].+randn(size(integrator.u[1])).*λ1.*exp(integrator.u[2]./2.0))
#affect1!(integrator) = (integrator.u[1] = λ1*integrator.u[1])    
jump = ConstantRateJump(rate,affect1!)
jump_prob = JumpProblem(prob,Direct(), jump)
sol = solve(jump_prob,SRIW1(), dt=dt, adaptive=false)
# get lnP at each tic
lnP = [sol(t)[1] for t in 0:dt:(Days-1)]
# get log price at end of trading days
z = zeros(TradingDays+1)
RV = zeros(TradingDays+1)
global DayofWeek = 0 # counter for day of week
global TradingDay = 0 # counter for trading days
global Day = 0
while TradingDay<=TradingDays
    # set day of week, and record if it's a trading day
    Day +=1
    DayofWeek +=1
    if DayofWeek<6
        TradingDay +=1 # advance trading day
        # compute realized measures
        for tic = 2:closing
            RV[TradingDay]+=(lnP[Day*tics+tic]-lnP[Day*tics+tic-1])^2.0
        end    
        z[TradingDay] = lnP[Day*tics+closing]
    end
    if DayofWeek==7 # restart the week if Sunday
        DayofWeek = 0
    end   
end
rets = z[2:end]-z[1:end-1]
RV = RV[2:end]

# checking the jump rate
t = sol.t
jump = t[2:end].==t[1:end-1] # find where time does not change
jump[1]=0 # there is alway a "jump" at beginning, drop it
println("number of jumps: ", sum(jump), " in ", TradingDays, " days")
println("empirical jump rate (?): ", sum(jump)/TradingDays)
println("nominal jump rate: ", λ0)
return z, jump, sol, rets, RV;
end
z, jump, sol, rets, RV = dgp();
t = sol.t;
t = t[2:end]
u = sol(t[jump]).u
jumpvals = [u[i][1] for i in 1:sum(jump)]
plot(rets, label="returns");
vline!(t[jump].*1000/1400, label="jumps")

