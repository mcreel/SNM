# SV model
These are results for the simple discrete time stochastic volatility model

## True parameter values
The true parameters used to simulate an artificial sample are
```
function TrueParameters()
    [exp(-0.736/2.0), 0.9, 0.363]
end
```

## Estimation results
Using a sample generated at these parameter values, estimation results are

![results](results.png)

The chain and marginal posteriors are

![chain](chain.png)


## Monte Carlo results
There are  two versions of the estimator available: one using a fixed weight matrix, and
one using a continuously updated weight matrix (see the do_cue option in src/MCMC.jl). 

Doing 500 Monte Carlo replications, by running ```mpirun -np 21 julia --project MonteCarlo.jl``` the following results were obtained for the coverage of confidence intervals defined by quantiles of the MCMC chain, for each of the seven estimated parameters:

### First, for the fixed weight matrix option: ###
![CIs](mcresults.png)

Some of these coverage values are statistically significantly different from what is expected for truly accurate confidence intervals, but most are not, and the departures from correct coverage are not large.

The Monte Carlo means, medians, and standard deviations are:
![rmses](mcresults2.png)
So, the estimator has extremely low bias, and thus, RMSE is essentially the same as the
std. dev.

### Second, for the CUE weight matrix option: ###
![CUECIs](mcresultsCUE.png)

These coverage results are improved, and in all cases are not significantly different from correct.

The Monte Carlo means, medians, and standard deviations are:
![CUErmses](mcresults2CUE.png)

These are very similar to the above results for the two-step version.

In conclusion, the CUE and two-step versions obtain the same results for bias and RMSE.
The CUE version gives entirely reliable confidence intervals. The CIs from the two-step
version are too broad in some instances, and may lead to type-II errors more often than
would the CUE version.

