# SNM

This archive contains the code for the examples reported in the paper "Inference
Using Simulated Neural Moments", Econometrics 2021, 9(4), 35; https://doi.org/10.3390/econometrics9040035  In Releases, you can find the exact code which was used in the paper. The code in the master branch of this archive is updated and improved, compared to the code used for the paper.

There is a registered Julia package [SimulatedNeuralMoments.jl](https://github.com/mcreel/SimulatedNeuralMoments.jl) which allows convenient use of the methods. The MN and SV examples in the paper are now presented in that package.

The DSGE and Jump Diffusion examples in the master branch of this archive use the most recent version of the SimulatedNeuralMoments package. These are two more interesting and research-relevant examples. The Jump Diffusion model results have been considerably improved compared to the results reported in the paper, by using a data rejection strategy to eliminate economically unrealistic simulations from the training data. See the JD directory for the updated results. 

A more recent paper "Constructing Efficient Simulated Moments Using Temporal Convolutional Networks" by Chassot and Creel, 2023, https://jldc.ch/uploads/2023_chassot_creel.pdf uses Temporal Convolutional Nets using the full sample as the input to the net, rather than a vector of statistics. This alternative version methodology has somewhat better performance, but is more demanding to use. The methods of this archive have quite good performance, and are fairly easy to use.
