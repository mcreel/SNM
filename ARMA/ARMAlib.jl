using Statistics, StatsBase, Random

function ARMAmodel(θ)
    N = 300
    burnin = 50
    α = θ[1]
    β = θ[2]
    σ = sqrt(θ[3])
    data = zeros(N)
    xlag = 0.0
    flag = 0.0
    @inbounds for i = 1:(N+burnin)
        f = σ*randn()
        x = α*xlag + f - β*flag
        xlag = x
        flag = f
        if i > burnin
            data[i-burnin] = x
        end    
    end   
    data
end

function auxstat(θ, reps)
    data = [ARMAmodel(θ, rand(1:Int64(1e12))) for i = 1:reps]  # reps draws of data
    auxstat.(data)
end

@views function auxstat(data)
    x = arma11(θ)
    s = std(x)
    c = cor(x[2:end,:],x[1:end-1])
    n = size(x,1)
    b, varb, u, junk1, rsq  = ols(x[2:end,:], x[1:end-1,:], silent=true)
    ϕ, varϕ, u, junk2, rsq2 = ols(u[2:end,:], u[1:end-1,:], silent=true)
    σ = std(u)
    sqrt(300.0).*vcat(s, c, b, varb, rsq, ϕ, varϕ, rsq2, σ, pacf(x,collect(1:4)))
end    

function TrueParameters()
    [0.95, 0.5, 1.0]
end    

function PriorSupport()
    lb = [-0.99, -0.99, 0.0001]
    ub = [0.99, 0.99, 4.0]
    lb,ub
end    

function PriorMean()
    lb, ub = PriorSupport()
    (ub - lb) ./ 2.0
end

function PriorDraw()
    lb, ub = PriorSupport()
    (ub - lb) .* rand(3) + lb
end

# prior just checks that we're in the bounds
function Prior(theta)
    lb, ub = PriorSupport()
    a = 0.0
    if(all((theta .>= lb) .& (theta .<= ub)))
        a = 1.0
    end
    return a
end

# from Econometrics
function ols(y::Array{Float64}, x::Array{Float64,2}; R=[], r=[], names="", vc="white", silent=false)
    n,k = size(x)
    if names==""
        names = 1:k
        names = names'
    end
    b, fit, e = lsfit(y,x)
    df = n-k
    sigsq = (e'*e/df)[1,1]
    xx_inv = inv(x'*x)
    ess = (e' * e)[1,1]
    xe = x.*e
    varb = xx_inv*xe'xe*xx_inv
    seb = sqrt.(diag(varb))
    seb = seb.*(seb.>1e-16) # round off to zero when there are restrictions
    t = b ./ seb
    tss = y .- mean(y)
    tss = (tss'*tss)[1,1]
    rsq = (1.0 - ess / tss)
    return b, varb, e, ess, rsq
end
