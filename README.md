# SNM

This archive contains the code for the examples reported in the paper "Inference
Using Simulated Neural Moments", Econometrics 2021, 9(4), 35; https://doi.org/10.3390/econometrics9040035  In Releases (over to the right), you can find the exact code which was used in the paper. The code in the master branch of this archive is updated and improved, compared to the code used for the paper.

There is a registered Julia package [SimulatedNeuralMoments.jl](https://github.com/mcreel/SimulatedNeuralMoments.jl) which allows convenient use of the methods. The MN and SV examples in the paper are now presented in that package.

The DSGE and Jump Diffusion examples in the master branch of this archive use the most recent version of the SimulatedNeuralMoments package. These are two more interesting and research-relevant examples. The Jump Diffusion model results have been considerably improved compared to the results reported in the paper, by using a data rejection strategy to eliminate economically unrealistic simulations from the training data. See the JD directory for the updated results. 

An earlier paper, upon which this one builds, is "Neural Nets for Indirect Inference" https://www.sciencedirect.com/science/article/pii/S2452306216300326 This paper explore using neural nets to map a vector of statistic using data generated from a structural model to the parameter vector that characterizes the structural model. 

A later paper, which extends the results of this one and the previous one, is "Constructing Efficient Simulated Moments Using Temporal Convolutional Networks" with J. Chassot (under review) https://jldc.ch/uploads/2023_chassot_creel.pdf This last paper does away with the need to specify statistics: the net is used to directly map the data from the model to the parameters that generated the data. This approach has the advantage of avoiding information loss due to a poor choice of statistics. The methods can rival the RMSE properties of the maximum likelihood estimator, for small and moderately sized samples.
