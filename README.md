# SNM
Release v1.0 is the code to accompany the working paper <a href=https://www.barcelonagse.eu/research/working-papers/inference-using-simulated-neural-moments>Inference using simulated neural moments</a> The code in the release allows estimating using plain and neural moments, and using as the criterion the full indirect likelihood (L in the paper) or the GMM form (H in the paper). 

The paper finds that the GMM-form works as well as the full form, so the code in master focuses on that. To use the code, first, set JULIA_NUM_THREADS. Then edit RunProject.jl to select one of the examples, by uncommenting the relevant lines. CD to the directory of the chosen example, start Julia, and execute include("../RunProject.jl"). Running the SV example using 10 threads, on a fairly old server, I can estimate the SV model in about 20 seconds. That does not count the time to train the net, but training the net is a process that is independent of the real data, so including its time as part of the time to estimate does not make much sense.

![example](https://github.com/mcreel/SNM/blob/master/abstract.png)



