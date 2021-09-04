using CSV, DataFrames, DelimitedFiles, DataFramesMeta
data = CSV.read("sp500.csv", DataFrame)
data = Matrix{Float64}(data[2:end,[:close, :rv, :bv]])
rets = 100.0*(log.(data[2:end,1]) - log.(data[1:end-1,1]))
rv = 100000.0 .*data[2:end,2]
bv = 100000.0 .*data[2:end,3]
writedlm("sp500.txt", [rets rv bv]) 
