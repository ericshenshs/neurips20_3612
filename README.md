The code outputs the figures in the NeurIPS 2020 paper with paper ID 3612,
*Asymptotic normality and confidence intervals for derivatives of 2-layers neural network in the random features model*.

Running `python3 run_plots.py` in the command line under the repository directory shall output the figures to the same directory. The figures are expected to appear in several minutes: 

- Figure 1 shall appear in 2-10 min. 
- Figure 2 are expected to appear in 5-40 min.
- The waiting time for Figure 1 and Figure 2 to appear depends on the number of iterations in the simulation and the computing power. It may take a long time to wait if the number of iterations is with the default value 300. Decreasing the number of iterations will significantly decrease the running time. 
- Figure 3 are expected to be plotted in 0-10 seconds. It is fast since no iteration is involved. 

The code has been tested under the following environment:

- Ubuntu 19.10 and Python 3.7.4 with packages: numpy, statsmodels, pylab, scipy, matplotlib, sympy.
