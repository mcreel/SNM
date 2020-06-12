using Econometrics, Statistics, Flux, Random, LinearAlgebra
using Base.Iterators
using BSON: @load
using BSON: @save

function Train(TrainingTestingSize)
    # basic configuration
    #find out how many parameters
    lb, ub = PriorSupport()
    nParams = size(lb,1)
    # size of training/testing
    TrainingProportion = 0.5
    Epochs = 1000 # passes through entire training set
    # read the training/testing data, and define the splits
    @load "cooked_data.bson" params statistics
    params = Float32.(params)
    statistics = Float32.(statistics)
    S = TrainingTestingSize # number of draws from prior
    trainsize = Int(TrainingProportion*S)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = statistics[1:trainsize, :]'
    xout = statistics[trainsize+1:end, :]'
    # define the neural net
    nStats = size(xin,1)
    model = Chain(
        Dense(nStats, 10*nParams, tanh),
        Dense(10*nParams, nParams)
    )
    θ = Flux.params(model)
    opt = ADAGrad()
    # Define the loss function
    # weight by inverse std. dev. of params in sample from prior, to put equal weight
    s = Float32.(std(params,dims=1)')
    #loss(x,y) = sqrt.(Flux.mse(model(x)./s,y./s))
    loss(x,y) = Flux.mse(model(x),y)
    # monitor training
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end

    # do the training
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 1024)];
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
            pred = model(xx)
            error = yy .- pred
            results = [pred;error]
            rmse = sqrt.(mean(error.^Float32(2.0),dims=2))
            println(" ")
            println("RMSE for model parameters ")
            prettyprint(reshape(round.(rmse,digits=3),1,nParams))
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
