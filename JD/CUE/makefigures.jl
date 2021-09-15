using Econometrics, DelimitedFiles, StatsPlots, MCMCChains

chain = readdlm("chain")
chn = Chains(chain,["μ", "κ","α","σ","ρ","λ0","λ1","τ"])
display(chn)
plot(chn)
savefig("chain.png")

p = npdensity(chain[:,1])
plot!(p, legend=false)
savefig(p, "mu.png")

p = npdensity(chain[:,2])
plot!(legend=false)
savefig(p, "kappa.png")

p = npdensity(chain[:,3])
plot!(p, legend=false)
savefig(p, "alpha.png")

p = npdensity(chain[:,4])
plot!(p, legend=false)
savefig(p, "sigma.png")

p = npdensity(chain[:,5])
plot!(p, legend=false)
savefig(p, "rho.png")

p = npdensity(chain[:,6])
plot!(p, legend=false)
savefig(p, "lambda0.png")

p = npdensity(chain[:,7])
plot!(p, legend=false)
savefig(p, "lambda1.png")

p = npdensity(chain[:,8])
plot!(p, legend=false)
savefig(p, "tau.png")
