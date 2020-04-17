function MSMsigma(θ, m, S)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = ILSNM_model(θ[:])
    end
    Σ = cov(ms)
    return Σ
end

function MSMmoments(θ, m, S)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = ILSNM_model(θ[:])
    end
    mbar = mean(ms,dims=1)[:]
    x = (m .- mbar)[:]
    return x
end


function MSMobj(θ, m, S)
    k = size(m,1)
    ms = zeros(S, k)
    Threads.@threads for s = 1:S
        @inbounds ms[s,:] = ILSNM_model(θ[:])
    end
    mbar = mean(ms,dims=1)[:]
    Σ = cov(ms)
    x = (m .- mbar)[:]
    #lnL =-0.5*logdet(Σ) - 0.5*x'*inv(Σ)*x
    lnL = x'*inv(Σ)*x
    obj = 500.0*lnL
    return obj
end

