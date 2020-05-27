using Pkg
Pkg.activate(".")
using Econometrics, DelimitedFiles, Statistics

files = [
    "baseline_GMM",
    "baseline_NN",
    "baseline_no_Jacobian"]


function getresults()
    for i = 1:3
        if i==1
            dir="SV/"
            θtrue = [exp(-0.736/2.0), 0.9, 0.363]
        elseif i==2
            dir="DPD/"
            θtrue = [0.6, 1.0, 2.0]
        else
            dir="ARMA/"
            θtrue = [0.95,0.5,1.0]
        end
        PrintDivider()
        println(dir)
        r = zeros(3,3)
        b = similar(r)
        c90 = similar(r)
        c95 = similar(r)
        c99 = similar(r)
        for j = 1:3
            d = readdlm(dir*files[j])
            error = d[:,1:3] .- θtrue'
            b[:,j] = mean(error, dims=1) ./θtrue'
            r[:,j] = sqrt.(mean(error.^2.0, dims=1)) ./θtrue'
            c99[:,j] = mean(d[:,4:6], dims=1)
            c95[:,j] = mean(d[:,7:9], dims=1)
            c90[:,j] = mean(d[:,10:12], dims=1)
        end
        println("rmse")
        display(round.(r,digits=3))
        println("bias")
        display(round.(b,digits=3))
        println("c90")
        display(round.(c90,digits=3))
        println("c95")
        display(round.(c95,digits=3))
        println("c99")
        display(round.(c99,digits=3))
    end    
end

getresults()



