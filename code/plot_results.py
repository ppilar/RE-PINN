# -*- coding: utf-8 -*-
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from punc.plots import *
from punc.utils_eval_results import *
from punc.utils_HMC import get_MC_samples, ResultsMC

#%%
def add_MC_metrics(res, res_mc):
    res.KL_gesges = np.zeros((res.pars['Nmodel'], res.pars['Nrun']))
    res.KL_dpar_gesges = np.zeros((res.pars['Nmodel'], res.pars['Nrun']))
    res.W_gesges = np.zeros((res.pars['Nmodel'], res.pars['Nrun']))
    res.W_dpar_gesges = np.zeros((res.pars['Nmodel'], res.pars['Nrun']))
    
    for jN in range(res.pars['Nrun']):
        #mc_samples, preds = get_MC_samples(res.pars, res.data[jN])
        #res_mc.store_run_results(res.data[jN], preds, mc_samples, jN)
        res.KL_gesges[:,jN], res.KL_dpar_gesges[:,jN] = calculate_MC_agreement(res, res_mc, jN, opt='KL')
        res.W_gesges[:,jN], res.W_dpar_gesges[:,jN] = calculate_MC_agreement(res, res_mc, jN, opt='W')
        
#%%
# fpath = '../results/res_ds1_N_ensemble/ds1_N_ensemble50/'
# fname = 'x1N50r5.dat'
# fpath_mc = '../results/res_ds1_N_ensemble/ds1_N_ensemble5/'
# fname_mc = 'x1N5r5.dat' 

# fpath= '../results/res_ds7_N_ensemble25_fn1.0/'
# fname = 'x7N25r5.dat'

fpath = '../results/res_ds102_N_ensemble50_repf_lf0.1/'
fname = 'x102N50r5.dat'

# fpath = '../results/res_ds5_N_train10v2_gap_fn1/'
# fname = 'x5N25r5.dat'





with open(fpath + fname, 'rb') as f:
    res = pickle.load(f)
    
    
#%%
try:
    if not 'fpath_mc' in locals():
        fpath_mc = fpath
        fname_mc = fname
    with open(fpath_mc + 'mc_' + fname_mc, 'rb') as f:
        res_mc = pickle.load(f) 
except:
    print('No MC data available.')
    res_mc = None
    
ppath = fpath + 'plots/'


#%%
#calculate MC metrics and update logL with value resulting from KDE approximation
calculate_metrics = True
if calculate_metrics:
    for jN in range(res.pars['Nrun']):
        res.calculate_res_stats(jN)
    if res_mc is not None:
        add_MC_metrics(res, res_mc)
# # #%%
# with open(fpath + fname,'wb') as f:
#     pickle.dump(res, f)
    
    
from punc.utils_HMC import get_MC_samples, ResultsMC
#%%
print_ensemble_comparison_table(res, res.pars['Nrun'], fpath=fpath, opt='flipped')

#%%
if not 't_opt' in res.pars:
    res.pars['t_opt'] = ''

for jN in range(res.pars['Nrun']): 
    jm_vec = res.pars['model_vec']
    dist_opt = 'median'
    #%% plot par distribution
    if False:
        if res.pars['dim'] == 1:
            dpar_buf = np.concatenate((np.expand_dims(res.xnp_gesges[:,:,jN,0,0],1), res.dpar_gesges[:,:,:,jN,-1]),1)
            if res_mc is not None:
                fig = pairwise_plot(res.pars, dpar_buf.swapaxes(1,2), mc_samples = res_mc.dpar_gesges[0,:,:,jN].T, plot_opt='', jm_vec = jm_vec)
            else:
                fig = pairwise_plot(res.pars, dpar_buf.swapaxes(1,2), plot_opt='', jm_vec = jm_vec)
                
        else:
            fig = pairwise_plot(res.pars, res.dpar_gesges[:,:,:,jN,-1].swapaxes(1,2), plot_opt='', jm_vec = jm_vec)
        fig.savefig(fpath+'plots/ldist_jN'+str(jN)+'.pdf', bbox_inches='tight')
    
    #%% plot f distribution    
    match res.pars['x_opt']:
        case 1:
            plist = ['f', 'l0']
            #plist = ['f', 's', 'l0']
            #plist = ['s']
        case 5:
            res.pars['ylim'] = [-0.8, 1.1]
            plist = ['f']
            plist = ['f', 's', 'l0', 'l1']
        case 7:
            plist = ['f', 's', 'l0', 'l1', 'l2', 'l3']
            #plist = ['f','l2']
        case 102:
            plist = ['f', 'l0']
    plot_distribution_and_samples(res, res_mc, jN, jm_vec, plist, fpath+'plots/', 'sdist_jN'+str(jN))

    
    
#%% overview plot
plot_overview_plot = False
if plot_overview_plot:
    jm, jN = 6, 3
    plot_overview(res, res_mc, jm, jN, ppath, 'overview')
    
  
#%% plot VI
plot_overview_plot_VI = False #load data at the end of script before running this cell
if plot_overview_plot_VI:
    jm, jN = 0, 3
    plot_overview_VI(res_VI, res_mc, jm, jN, ppath, 'overview_VI')

    
    

#%%
from punc.plots import axplot_f_mc

plot_ds1_front_figure = False
if plot_ds1_front_figure:    
    jm, jN = 5, 0
    plot_frontpage_ds1(res, res_mc, jm, jN, ppath, 'frontpage')
    


#%%

jm, jN = 0, 0
plot_ds102_data = True
if plot_ds102_data:
    plot_data_ds102(res, jm, jN, fpath + 'plots/', 'data_jN'+str(jN))
    
#%%
plot_ds102_samples = True
if plot_ds102_samples:
    jN = 0
    jm_vec = [0, 1, 2, 4, 5, 6]
    plot_samples_ds102(res, jm_vec, jN,fpath + 'plots/', 'samples_jN'+str(jN))

                
    
#%%
plot_ds1_Ncomp = False
if plot_ds1_Ncomp:
    #%% combine multiple par settings
    folder, par = '../results/res_ds1_N_ensemble/', 'N_ensemble'
    res_ges, par_vals, res_mc_ges = gather_results(folder, par)
    metric_names, metric_values, digits, minmax = get_standard_metrics(res_ges)
    model_captions = get_par_captions(par, par_vals)
    #%%
    res_ges[0].pars['model_vec'] = res_ges[0].pars['model_vec'][:-1]
    #%%
    jN = 4
    from punc.plots import print_comparison_table
    for jm in res_ges[0].pars['model_vec']:
        print_comparison_table(metric_names, metric_values[:,:,jm,:], model_captions, digits, minmax, fpath=folder, fname='stats_jm'+str(jm))
        plot_model_results_vs_par(res_ges, res_mc_ges, model_captions, jm, jN, dist_opt = 'median', ppath='../plots/ensembles/')

    #%%
    plot_parameter_comparison(res_ges, par, par_vals, metric_names, metric_values, ppath='../plots/ensembles/')



#%%
from VI.pibnn.utils_bnn import ResultsVI
with open(fpath_mc + 'VI_' + fname_mc, 'rb') as f:
    res_VI = pickle.load(f) 
        
