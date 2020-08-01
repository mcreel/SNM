using Plots
using Econometrics
include("JDlib.jl")
θ = TrueParameters()
#θ = PriorDraw()
rets, RV, MedRV, ret0, Monday = dgp(θ,reps)
rets, Volatility, jumptimes, RV, MedRV, ret0, Monday = dgp(θ);
stats = auxstat(θ,1)

#=
#Ret0 = lsfit(abs.(Ret0),[ones(1000) Monday])[3]
plot(layout=(4,1))
p1 = plot(rets, label="rets")
vline!(jumptimes, label="jumps", linealpha=0.5)
p2 = plot(Ret0,label="ret0")
vline!(jumptimes, label="jumps", linealpha=0.5 )
p3 = plot(RV, label="RV")
vline!(jumptimes, label="jumps", linealpha=0.5)
p4 = plot(MedRV, label="MedRV")
vline!(jumptimes, label="jumps", linealpha=0.5)
plot(p1,p2, p3, p4)
=#


