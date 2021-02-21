using MCMCChains, StatsPlots, DelimitedFiles, Econometrics
chain = readdlm("chain")
chain = chain[500:end,:] # drop burnin period
chn = Chains(chain)
display(chn)
plot(chn)
#=
savefig("chain.png")
savefig(npdensity(chain[:,1]), "mu.png")
savefig(npdensity(chain[:,2]), "kappa.png")
savefig(npdensity(chain[:,3]), "alpha.png")
savefig(npdensity(chain[:,4]), "sigma.png")
savefig(npdensity(chain[:,5]), "rho.png")
savefig(npdensity(chain[:,6]), "lambda0.png")
savefig(npdensity(chain[:,7]), "lambda1.png")
savefig(npdensity(chain[:,8]), "tau.png")
=#
