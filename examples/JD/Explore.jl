using Plots
using Econometrics
include("JDlib.jl")
θ = TrueParameters()
#rets, Volatility, jumptimes, RV, MedRV, RVinter, MedRVinter, Monday = dgp(θ);

#RVinter = lsfit(RVinter,[ones(1000) Monday])[3]
plot(layout=(2,1))
p1 = plot(RV)
vline!(jumptimes, label="jumps")
p2 = plot(RVinter)
vline!(jumptimes, label="jumps")
plot(p1,p2)



