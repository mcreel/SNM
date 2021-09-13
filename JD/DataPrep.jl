using CSV, DataFrames, DataFramesMeta, Plots, Dates, DelimitedFiles
# download and unzip the data file, from
# https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip
# read the data into a DataFrame
data = CSV.read("oxfordmanrealizedvolatilityindices.csv",DataFrame)
# select the desired columns
data = data[:,[:Column1, :Symbol, :close_price, :rv10, :bv]]
# set desired names
rename!(data, :Column1 => :Date)
rename!(data, :rv10 => :rv)
# select desired index
sp500 = @subset(data, :Symbol .== ".SPX")
# drop times from date by taking the leading 10 characters (yyyy-mm-dd)
sp500.Date = [sp500.Date[i][1:10] for i = 1:size(sp500,1)]
# compute percentage returns
sp500.rets = vcat(0.0,[100.0 *(log(sp500.close_price[i]) - log(sp500.close_price[i-1])) for i=2:size(sp500,1)])
# scale volatility measures to be compatible with percentage returns (log price mult. by 100)
sp500.rv = 10000.0 .* sp500.rv
sp500.bv = 10000.0 .* sp500.bv
# select the date range
sp500 = @subset(sp500, :Date .>= "2013-12-17") # these dates give 1000 obs.
sp500 = @subset(sp500, :Date .<= "2017-12-05")
# write out variables to plain text file
data = sp500[:,[:rets, :rv, :bv]]
data = Matrix{Float64}(data)
CSV.write("sp500.csv", sp500)
#writedlm("sp500.txt", data)
# some plots
#plot(sp500.Date, sp500.rets, tickfontsize=5, legend=false)
#savefig("returns.png")
#plot(sp500.Date, [sp500.rv, sp500.bv], tickfontsize=5, label=["rv" "bv"])
#savefig("volatility.png")

