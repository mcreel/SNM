using Statistics, Econometrics
function AnalyzeMSM(θ, se)
    nParams = size(θ,1)
    inci01 = zeros(nParams)
    inci05 = zeros(nParams)
    inci10 = zeros(nParams)
    lower = zeros(nParams)
    upper = zeros(nParams)
    for i = 1:nParams
        lower[i] = θ[i] - 2.576*se[i]
        upper[i] = θ[i] + 2.576*se[i]
        inci01[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
        lower[i] = θ[i] - 1.96*se[i]
        upper[i] = θ[i] + 1.96*se[i]
        inci05[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
        lower[i] = θ[i] - 1.645*se[i]
        upper[i] = θ[i] + 1.645*se[i]
        inci10[i] = θtrue[i] >= lower[i] && θtrue[i] <= upper[i]
    end
    return vcat(inci01[:], inci05[:], inci10[:])
end   

