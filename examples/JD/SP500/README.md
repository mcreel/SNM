This directory holds an example of estimation of the jump diffusion model 
![model](https://github.com/mcreel/SNM/blob/master/examples/JD/SimulationEstimation/model.png)

using data on S&P500 returns from Dec. 2013 - Dec. 2017. Log returns, p, is assumed to be measured with a
N(0,\tau^2) error.

The results are in the .png files. For example, for the parameter tau, the standard deviation
of measurement error in observed returns, the marginal posterior is
is in the following figure. When tau<0, there is no measurement error. We see that there is strong evidence in favor of measurement error being a factor.
![tau](https://github.com/mcreel/SNM/blob/master/examples/JD/SP500/tau.png)

