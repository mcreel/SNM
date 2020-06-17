# SNM
This is a project to reduce the dimension of statistics used for Approximate Bayesian Computing or the method of simulated moments though use of neural nets. The project allows for creation and training of the neural net, and for calculation of the neural moments, given the trained net. It also provides the large sample indirect likelihood function of the neural moments, which can be used to sample from the posterior, using MCMC (simple version provided in the project), SMC (not provided), or other methods. The results reported in the paper and below are a product of two main features: the use of neural moments to reduce the dimension of the summary statistics, and the use of the indirect likelihood function as the criterion or distance measure. The code for these two features is [here](https://github.com/mcreel/SNM/blob/master/src/SNM.jl).

The project allows for Monte Carlo investigation of the performance of estimators and the reliability of confidence intervals obtained from the quantiles samples from the posterior distribution.

The project is motivated by results in the working paper <a href=https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments>Inference using simulated neural moments</a> The code in the WP branch of this archive allows for replication of the results in that paper. The master branch builds on the results of the paper to focus on the best performing methods.

# Worked example
The following is an explanation of how to use the code in the master branch.

1. git clone the project into a directory. Go to that directory, set the appropriate number of Julia threads, given your hardware, e.g. ```export JULIA_NUM_THREADS=10```
2. start Julia, and do ```]activate .``` to set up the dependencies correctly. This will take quite a while the first time you do it, as the project relies on a number of packages.
3. do ```include("RunProject.jl)```  to run a Monte Carlo study of simple example based on a mixture of normals.

The mixture of normals model (see the file [MNlib.jl](https://github.com/mcreel/SNM/blob/master/examples/MN/MNlib.jl) for details) draws statistics using the function
```
function auxstat(θ)
    n = 1000
    μ_1, μ_2, σ_1, σ_2, prob = θ
    d1=randn(n).*σ_1 .+ μ_1
    d2=randn(n).*σ_2 .+ μ_2
    ps=rand(n).<prob
    data=zeros(n)
    data[ps].=d1[ps]
    data[.!ps].=d2[.!ps]
    r=0:0.1:1
    sqrt(Float64(n)).* quantile.(Ref(data),r)
end
```    

So, there are five parameters, and 11 summary statistics. Samples of 1000 observations are used to compute the statistics. The "true" parameter values we will use to evaluate performance and confidence interval coverage are from
```
function TrueParameters()
    [1.0, 0.0, 0.2, 2.0, 0.4]
end
```    

When we run ```RunProject()```, as above, we obtain output similar to the following results, for 1000 Monte Carlo replications:
![MCresults](https://github.com/mcreel/SNM/blob/master/MCresults.png)

The parameters are estimated with little bias, and good precision, and confidence interval coverages are close to the nominal levels, for each of the 5 parameters.


4. do ```include("examples/MN/EstimateMN.jl")``` to do a single estimation of the mixture of normals model. We can visualize the posterior densities for the parameters, and the tail quantiles which define a 90% confidence interval.

![MNp1](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp1.png)
![MNp2](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp2.png)
![MNp3](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp3.png)
![MNp4](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp4.png)
![MNp5](https://github.com/mcreel/SNM/blob/master/examples/MN/MNp5.png)

