using Pkg
Pkg.activate("../")
using Econometrics, DelimitedFiles, Statistics
θtrue = [0.6, 1.0, 2.0]
files = [
    "baseline_NN",
    "no_Jacobian_NN"];
for j = 1:2
    d = readdlm(files[j])
    error = d[:,1:3] .- θtrue'
    b = mean(error, dims=1) ./θtrue'
    r = sqrt.(mean(error.^2.0, dims=1)) ./θtrue'
    c99 = mean(d[:,10:12], dims=1)
    c95 = mean(d[:,13:15], dims=1)
    c90 = mean(d[:,16:18], dims=1)
    names = ["rmse", "bias", "c90","c95","c99"]
    prettyprint([r' b' c90' c95' c99'],names,"")
end    


