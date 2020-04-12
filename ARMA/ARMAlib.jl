# simple ARMA(1,1) model from Fiorentini et al. 2018, "A spectral EM algorithm for dynamic factor models",
# page 256
using Econometrics, Statistics, Random

function arma11(θ)
    N = 300
    burnin = 50
    α = θ[1]
    β = θ[2]
    σ = sqrt(θ[3])
    data = zeros(N,1)
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

function ILSNM_model(θ)
    x = arma11(θ)
    s = std(x)
    #x = x ./s
    c = cor(x[2:end,:],x[1:end-1])
    n = size(x,1)
    b, junk, u, junk, junk = tsls(x[4:end,:], x[3:end-1,:],[x[1:end-3,:] x[2:end-2]]; silent=true)
    ϕ, junk, u = lsfit(u[2:end,:], u[1:end-1,:])
    σ = std(u)
    b2, junk, u = lsfit(x[2:end,:], x[1:end-1,:])
    ϕ2, junk, u = lsfit(u[2:end,:], u[1:end-1,:])
    σ2 = std(u)
#    vcat(std(x), c, b, ϕ, σ,b2, ϕ2, σ2)#, pacf(x,collect(1:4)), autocor(x,collect(1:4)))
    vcat(c, b, ϕ, σ, b2, ϕ2, σ2, pacf(x,collect(1:4)))
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

