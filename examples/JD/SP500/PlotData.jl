using DelimitedFiles, Plots, Dates
d = readdlm("SP500.txt")
d = d[2:end,:]
date = Date.(d[:,1],"y-m-d")
plot(date, d[:,2], legend=false)
savefig("returns.svg")
plot(date, d[:,3], label="RV")
plot!(date, d[:,4], label="BV")
savefig("volatility.svg")
