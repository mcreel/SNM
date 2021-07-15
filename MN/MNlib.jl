using Statistics, StatsBase, Random

function MNmodel(θ, rndseed=1234)
    Random.seed!(rndseed)
    n = 1000
    μ1, μ2, σ1, σ2, prob = θ
    d1=randn(n).*σ1 .+ μ1
    d2=randn(n).*(σ1+σ2) .+ (μ1 - μ2) # second component lower mean and higher variance
    ps=rand(n).<prob
    data=zeros(n)
    data[ps].=d1[ps]
    data[.!ps].=d2[.!ps]
    return data
end    

function auxstat(θ, reps)
    data = [MNmodel(θ, rand(1:Int64(1e12))) for i = 1:reps]  # reps draws of data
    auxstat.(data)
end

function auxstat(data)
    r = 0.0 : 0.1 : 1.0
    sqrt(Float64(size(data,1)))*vcat(mean(data), std(data), skewness(data), kurtosis(data), quantile.(Ref(data),r))
end 

function TrueParameters()
    [1.0, 1.0, 0.2, 1.8, 0.4] # first component N(1,0.2) second component N(0,2)
end    

function PriorSupport()
    lb = [0.0, 0.0, 0.0, 0.0, 0.05] # there is always at least 5% prob for each component  
    ub = [3.0, 3.0, 1.0, 3.0, 0.95] 
    lb,ub
end    

function InSupport(θ)
    lb,ub = PriorSupport()
    all(θ .>= lb) & all(θ .<= ub)
end

function PriorMean()
    lb,ub = PriorSupport()
    (ub - lb) ./ 2.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    θ = (ub-lb).*rand(size(lb,1)) + lb
end    

function Prior(θ)
    lb,ub = PriorSupport()
    a = 0.0
    if (all(θ .>= lb) & all(θ .<= ub))
        a = 1.0
    end
    return a
end


