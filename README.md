# SNM
This is a project to reduce the dimension of statistics used for Approximate Bayesian Computing or the method of simulated moments though use of neural nets. The project allows for creation and training of the neural net, and for calculation of the neural moments, given the trained net. It also provides the large sample indirect likelihood function of the neural moments, which can be used to sample from the posterior, using MCMC (provided) or SMC (not provided).

The project allows for Monte Carlo investigation of the performance of estimators and the reliability of confidence intervals obtained from the quantiles samples from the posterior distribution.

The project is motivated by results in the working paper <a href=https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments>Inference using simulated neural moments</a> The code in the WP branch of this archive allows for replication of the results in that paper. The master branch builds on the results of the paper to focus on the best performing methods.

# Worked example
The following is an explanation of how to use the code in the master branch.

1. git clone the project into a directory. Go to that directory, set the appropriate number of Julia threads given your hardware (e.g. ```export JULIA_NUM_THREADS=10```)
2. start Julia, and do ```activate .``` to set up the dependencies correctly. This will take quite a while, as the project relies on a number of packages.
3. do ```include("RunProject()```  to run a simple example based on a mixture of normals. 

Here is an explanation of the contents of ```RunProject.jl```




