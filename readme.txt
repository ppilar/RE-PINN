To perform experiments, run the 'punc.py' file. Settings can be specified in the 'input.py' file. The file located in the 'results' folder will be used.

Results of one experiment (with potentially multiple runs) can be plotted with the 'plot_results.py' file. This will als produce text files with statistics given in the Tables in the paper.

To run multiple experiments with different settings, use the 'master.py' file. A new folder will be created for each setting, if it does not exist already. The 'input.py' file from the 'results' folder will be used.

To run experiments on the advection data, first download the '1D_Advection_Sols_beta0.2.npy' file from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2986, and save it in the folder 'data'.

Run the files master_ds1, master_ds5, master_ds7, master_ds102 to perform the experiments for the exponential equation, the damped harmonic oscillator, the Lotka-Volterra equations, and the advection equation. The results will be saved in the 'results' folder.

Plots can be created by running 'plot_results.py'. At the beginning of the file, specify path and name of the results file to be selected.

To run Monte Carlo, use the 'run_HMC.py' file. Specify path and name of the results for which the MC baseline should be created.

With the file 'run_VI.py', the variational inference can be performed. Specify path and name of the results (i..e, the data) for which VI should be performed.