using SimulatedNeuralMoments, Statistics, DelimitedFiles
results = readdlm("mcresultsCUE.txt")
sofar = 500
println("__________ replication: ", sofar, "_______________")
clabels = ["99%", "95%", "90%"]
prettyprint(reshape(mean(results[1:sofar,4:end],dims=1),3,3),clabels)
dstats(results[1:sofar,1:3])

