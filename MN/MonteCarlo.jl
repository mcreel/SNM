# run this using mpirun -np X julia --proj MonteCarlo.jl
# where X is a divisor of 500, plus 1. X should also be less than
# or equal to the physical cores of your computer.
using SimulatedNeuralMoments, Flux, MPI, DelimitedFiles
using BSON:@save
using BSON:@load

include("MNlib.jl") # contains the functions for the DSGE model
include("Analyze.jl")
include("montecarlo.jl")

function Wrapper()
    lb, ub = PriorSupport()
    model = SNMmodel("MN example", lb, ub, InSupport, Prior, PriorDraw, auxstat)
    @load "neuralmodel.bson" nnmodel nninfo # use this to load a trained net
    data = MNmodel(TrueParameters(), rand(1:Int64(1e12)))
    m = NeuralMoments(auxstat(data), model, nnmodel, nninfo)
    @time chain, junk, junk = MCMC(m, 5500, model, nnmodel, nninfo; verbosity=false, do_cue = false)
    Analyze(chain[501:end,:])
end

function Monitor(sofar, results)
    nparams = size(TrueParameters(),1)
    if mod(sofar,1) == 0
        println("__________ replication: ", sofar, "_______________")
        clabels = ["99%", "95%", "90%"]
        prettyprint(reshape(mean(results[1:sofar,nparams+1:end],dims=1),nparams,3),clabels)
        dstats(results[1:sofar,1:nparams])
        if size(results,1)==sofar
            writedlm("mcresults.txt", results)
        end    
    end
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    reps = 500
    nparams = size(TrueParameters(),1)
    n_returns = 4*nparams 
    pooled = 1
    montecarlo(Wrapper, Monitor, comm, reps, n_returns, pooled)
    MPI.Finalize()
end
@time main()


