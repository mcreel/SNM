This directory holds an example of estimation of the jump diffusion model using artificial data generated at the parameter values
```function TrueParameters()
    μ = 0.02
    κ = 0.2
    α = 0.3
    σ = 0.7
    ρ = -0.7
    λ0 = 0.005 # jump rate per day
    λ1 = 4.0 # scaling factor st. dev. of jumps
    τ = 0.005
    return [μ, κ, α, σ, ρ, λ0, λ1, τ]
end```

The results are in in *.svg files. For example, for the parameter kappa, the marginal posterior is
