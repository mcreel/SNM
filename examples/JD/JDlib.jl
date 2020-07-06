using DifferentialEquations, Plots

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
Days = Int(TradingDays*7/5)  # calendar days
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
λ0 = 1.0 # jump rate
λ1 = 3.0 # scaling factor st. dev. of jumps
σme = 0.05 # standard dev of measurement error in returns
u0 = [0;α]
prob = Diffusion(μ0, μ1, κ, α, σ, ρ, u0, (0.0,Days))

## jump in price
rate(u,p,t) = λ0
# jump is normal with st. dev. equal to λ1 times current st. dev.
affect1!(integrator) = (integrator.u[1] = integrator.u[1].+randn(size(integrator.u[1])).*λ1.*exp(integrator.u[2]./2.0))
jump = ConstantRateJump(rate,affect1!)
jump_prob = JumpProblem(prob,Direct(), jump)
sol = solve(jump_prob,SRIW1(), dt=dt, adaptive=false)

# get log price at end of trading days
z = zeros(TradingDays+1)
global j = 0 # counter for day of week
global k = 0 # counter for trading days
for i = 0:(Days)
    # set day of week, and record if it's a trading day
    j +=1
    if j<6
        k +=1 # advance trading day
        z[k]=(sol.u)[i*tics+closing][1]
    end
    if j==7 # restart the week if Sunday
        j = 0
    end   
end
z[2:end]-z[1:end-1] + σme*randn(TradingDays) # returns are diff of log price
end
z = dgp()

plot(z)
