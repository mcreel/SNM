using Econometrics, Statistics, Flux, Random, LinearAlgebra
using Base.Iterators
using BSON: @load
using BSON: @save

# smoothed QR objective
function SmoothQRobj(yhat, y, τ, α=0.02)
    ξ = y .- yhat
    sum(τ.*ξ .+ α.*log.(1.0 .+ exp.(-1.0 .* ξ./α)))
end

function Train(TrainingTestingSize)
    # find out how many parameters
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    # size of training/testing
    TrainingProportion = 0.5
    Epochs = 1000 # passes through entire training set
    @load "cooked_data.bson" params statistics
    params = Float32.(params)
    statistics = Float32.(statistics)
    S = TrainingTestingSize # number of draws from prior
    trainsize = Int(TrainingProportion*S)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = statistics[1:trainsize, :]'
    xout = statistics[trainsize+1:end, :]'
    # model
    nStats = size(xin,1)
    model = Chain(
        Dense(nStats,3*nStats, tanh),
        Dense(3*nStats,3*nParams, tanh),
        Dense(3*nParams, nParams)
    )
    θ = Flux.params(model)
    opt = AdaMax()
    # weight by inverse std. dev. of params, to put equal weight
    s = Float32.(std(params,dims=1)')
    # Select the objective here: ordinary regression or (smoothed) quantile regression
    rmse(x,y) = sqrt.(Flux.mse(model(x)./s,y./s)) 
    # set the weight relative to RMSE
    weight = 0.01*rmse(xin,yin)/norm(θ) 
    @show weight 
    loss(x,y) = sqrt.(Flux.mse(model(x)./s,y./s)) .+ weight.*norm(θ)
    
    #τ = 0.5  # choose the quantile you want
    #loss(x,y) = QRobj(model(x),y, τ)
    #loss(x,y) = SmoothQRobj(model(x)./s,y./s, τ)
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 128)];
    for i = 1:Epochs
        Flux.train!(loss, θ, batches, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            @save "best.bson" model
            xx = xout
            yy = yout
            println("________________________________________________________________________________________________")
            monitor(i)
            pred = model(xx) # map pred to param space
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
            println(" ")
            println("RMSE for model parameters ")
            prettyprint(reshape(round.(rmse,digits=3),1,3))
            println(" ")
            println("dstats prediction of parameters:")
            dstats(pred');
            println(" ")
            println("dstats prediction error:")
            dstats(error');
        end
    end
    return nothing
end 
