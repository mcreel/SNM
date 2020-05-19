using Pkg
Pkg.activate("../")
using Econometrics, DelimitedFiles, Statistics
θtrue = [exp(-0.736/2.0), 0.9, 0.363]
files = [
    "baseline_NN",
    "baseline_raw",
    "baseline_no_Jacobian"];
for j = 1:3
    d = readdlm(files[j])
    error = d[:,1:3] .- θtrue'
    b = mean(error, dims=1) ./θtrue'
    r = sqrt.(mean(error.^2.0, dims=1)) ./θtrue'
    c99 = mean(d[:,4:6], dims=1)
    c95 = mean(d[:,7:9], dims=1)
    c90 = mean(d[:,10:12], dims=1)
    names = ["rmse", "bias", "c90","c95","c99"]
    prettyprint([r' b' c90' c95' c99'],names,"")
end    


