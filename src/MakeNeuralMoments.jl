# makes the training and testing data

using Econometrics, StatsBase, Statistics, Flux, Random, LinearAlgebra
using BSON: @save
using Base.Iterators

function MakeNeuralMoments(auxstat, S)
    data = 0.0
    datadesign = 0.0
    lb,ub = PriorSupport()
    nParams = size(lb,1)
    # training and testing
    W = auxstat(PriorMean(),1)' # draw the raw statistics
    data = zeros(S,size(vcat(lb, W),1))
    @inbounds Threads.@threads for s = 1:S
        ok = false
        θ = lb # initialize
        z = 0.0
        while !ok
            θ = PriorDraw()
            z = auxstat(θ,1)'
            ok = any(isnan.(z))==false
            if !ok println("warning: NaN in MakeNeuralMoments, retry"); end
        end    
        data[s,:] = vcat(θ, z)
    end
    # remove NaNs
    params = data[:,1:nParams]
    statistics = data[:,nParams+1:end]
    # transform stats to robustify against outliers
    q50 = zeros(size(statistics,2))
    q01 = similar(q50)
    q99 = similar(q50)
    iqr = similar(q50)
    for i = 1:size(statistics,2)
        q = quantile(statistics[:,i],[0.01, 0.25, 0.5, 0.75, 0.99])
        q01[i] = q[1]
        q50[i] = q[3]
        q99[i] = q[5]
        iqr[i] = q[4] - q[2]
    end
    transform_stats_info = (q01, q50, q99, iqr) 
    statistics = TransformStats(statistics, transform_stats_info)
    # train net
    # size of training/testing
    TrainingProportion = 0.5
    Epochs = 1000 # passes through entire training set
    params = Float32.(params)
    s = std(params, dims=1)'
    statistics = Float32.(statistics)
    trainsize = Int(TrainingProportion*S)
    yin = params[1:trainsize, :]'
    yout = params[trainsize+1:end, :]'
    xin = statistics[1:trainsize, :]'
    xout = statistics[trainsize+1:end, :]'
    # define the neural net
    nStats = size(xin,1)
    NNmodel = Chain(
        Dense(nStats, 10*nParams, tanh),
        Dense(10*nParams, nParams)
    )
    opt = ADAGrad() # the optimizer 
    loss(x,y) = Flux.mse(NNmodel(x)./s,y./s) # Define the loss function
    # monitor training
    function monitor(e)
        println("epoch $(lpad(e, 4)): (training) loss = $(round(loss(xin,yin); digits=4)) (testing) loss = $(round(loss(xout,yout); digits=4))| ")
    end
    # do the training
    bestsofar = 1.0e10
    pred = 0.0 # define it here to have it outside the for loop
    batches = [(xin[:,ind],yin[:,ind])  for ind in partition(1:size(yin,2), 1024)];
    bestmodel = 0.0
    for i = 1:Epochs
        Flux.train!(loss, Flux.params(NNmodel), batches, opt)
        current = loss(xout,yout)
        if current < bestsofar
            bestsofar = current
            bestmodel = NNmodel
            xx = xout
            yy = yout
            println("________________________________________________________________________________________________")
            monitor(i)
            pred = NNmodel(xx)
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
    NNmodel = bestmodel
    @save "neural_moments.bson" NNmodel transform_stats_info
    return nothing
end
