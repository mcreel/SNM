# CUE likelihood using net
function LL(θ, m, S, model, info, useJacobian=true)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = Float64.(model(transform(ILSNM_model(θ)', info)'))
    end
    mbar = mean(ms,dims=1)[:]
    Σ = cov(ms)
    x = (m .- mbar)[:]
    lnL = try
        if useJacobian
            lnL =-0.5*logdet(Σ) - 0.5*x'*inv(Σ)*x
        else
            lnL = -0.5*x'*inv(Σ)*x
        end    
    catch
        lnL = -Inf
    end    
    return lnL
end

# CUE likelihood without net
function LL(θ, m, S, useJacobian=true)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = ILSNM_model(θ)
    end
    mbar = mean(ms,dims=1)[:]
    Σ = cov(ms)
    x = (m .- mbar)[:]
    lnL = try
        if useJacobian
            lnL =-0.5*logdet(Σ) - 0.5*x'*inv(Σ)*x
        else
            lnL = -0.5*x'*inv(Σ)*x
        end    
    catch
        lnL = -Inf
    end    
    return lnL
end

# estimate covariance
function EstimateΣ(θ, m, S, model, info)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = Float64.(model(transform(ILSNM_model(θ)', info)'))
    end
    Σ = cov(ms)
end

# GMM-form quasi-likelihood using net and fixed estimate of Σ
function LL_with_fixed_Σ(θ, m, S, model, info, invΣ::Array{Float64})
    mbar = zeros(size(m))
    Threads.@threads for s = 1:S
        mbar .+= Float64.(model(transform(ILSNM_model(θ)', info)'))
    end
    x = m - mbar/S
    lnL = -0.5*dot(x,invΣ*x)
end

