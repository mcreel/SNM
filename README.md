# SNM

This archive contains the code for the examples reported in the paper "Inference
Using Simulated Neural Moments", Econometrics 2021, 9(4), 35; https://doi.org/10.3390/econometrics9040035

There is a registered Julia package ![SimulatedNeuralMoments.jl](https://github.com/mcreel/SimulatedNeuralMoments.jl) which allows convenient use of the methods. The code in the master branch of this archive uses the master branch of this package for the examples.

Here's the abstract of the paper:
![abstract](abstract.png)

Here are 95% confidence interval coverage results for the test models. The Z (CUE) results are for the method that the paper proposes.

Stochastic volatility and ARMA models:
![SVARMA](SVARMA.png)

Mixture of normals and DSGE models:
![MNDSGE](MNDSGE.png)
