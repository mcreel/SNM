using Distributions, KissABC

function ILSNM_model(θ)
    n = 5000
    data=mixture_model(θ,5000)
    sqrt(Float64(n)).*aux_stat(data)
end    


function mixture_model(P,N)
    μ_1, μ_2, σ_1, σ_2, prob=P
    d1=randn(N).*σ_1 .+ μ_1
    d2=randn(N).*σ_2 .+ μ_2
    ps=rand(N).<prob
    data=zeros(N)
    data[ps].=d1[ps]
    data[.!ps].=d2[.!ps]
    data
end

function aux_stat(data)
    r=0:0.01:1
    quantile.(Ref(data),r)
end



function PriorSupport()
    lb = [0.0, -2.0, 0.0, 0.0, 0.0]
    ub = [3.0, 2.0, 1.0, 4.0, 1.0]
    lb,ub
end    

function PriorMean()
    lb,ub = PriorSupport()
    (ub - lb) ./ 2.0
end

function Prior(θ)
    lb,ub = PriorSupport()
    a = 0.0
    if (all(θ .>= lb) & all(θ .<= ub))
        a = 1.0
    end
    return a
end


