import numpy as np
import torch
import pickle
#import scipy.linalg as splinalg
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import wasserstein_distance

from .plots import *

#calculate metrics that were added later; should eventually be removed
def calculate_res_stats(res):  
    if not hasattr(res, 'logL_dpar_ensemble_gesges') and res.pars['learn_rhs'] == 0:
        res.logL_dpar_ensemble_gesges = np.zeros([res.pars['Nmodel'],  res.pars['Nrun']])
        res.ABS_dpar_ensemble_gesges = np.zeros([res.pars['Nmodel'],  res.pars['Nrun']])
        
        for jN in range(res.pars['Nrun']):
            for jm in res.pars['model_vec']:
                for jl in range(len(res.pars['dpar'])):
                    print(jN, jm, jl)
                    buf_dpar = torch.tensor(res.dpar_gesges[jm,jl,:,jN,-1]).float()
                    dpar_dist = torch.distributions.Normal(buf_dpar.mean(), buf_dpar.std())
                    
                    res.ABS_dpar_ensemble_gesges[jm,jN] += ((res.pars['dpar'][jl] - buf_dpar.mean()).abs().numpy())/len(res.pars['dpar'])
                    res.logL_dpar_ensemble_gesges[jm,jN] += (dpar_dist.log_prob(torch.tensor(res.pars['dpar'][jl])).item())/len(res.pars['dpar'])
            
def get_Gaussian_KL(m1, m2, s1, s2):
    return np.log(s2/s1) + (s1**2 + (m1-m2)**2)/(2*s2**2) - 0.5

def calculate_MC_agreement(res, res_mc, jN, opt='KL'):    
    KL, KL_dpar = np.zeros((2, res.pars['Nmodel']))
    
    buf_mc = res_mc.xnp_gesges[0,:,jN,:,0]
    means_mc = buf_mc.mean(0)#.numpy()
    stds_mc = buf_mc.std(0)#.numpy()
    
    # means_l_mc = mc_buf[:,1:].mean(0).numpy()
    # stds_l_mc = mc_buf[:,1:].std(0).numpy()
    buf_l_mc = res_mc.dpar_gesges[0,1:,:,jN].T
    means_l_mc = buf_l_mc.mean(0)#.numpy()
    stds_l_mc = buf_l_mc.std(0)#.numpy()
    
    for j, jm in enumerate(res.pars['model_vec']):
        buf = res.xnp_gesges[jm,:,jN,:,0] #TODO: adjust for high-D output
        means = buf.mean(0)
        stds = buf.std(0)
        
        buf_l = res.dpar_gesges[jm,:,:,jN,-1].T
        means_l = buf_l.mean(0)
        stds_l = buf_l.std(0)
    
    
        if opt == 'KL':
            KL[jm] = get_Gaussian_KL(means, means_mc, stds, stds_mc).mean()
            KL_dpar[jm] = get_Gaussian_KL(means_l, means_l_mc, stds_l, stds_l_mc).mean()
        elif opt == 'W':
            for k in range(buf.shape[1]):
                KL[jm] += wasserstein_distance(buf_mc[:,k], buf[:,k])/buf.shape[1]
            for k in range(buf_l.shape[1]):
                KL_dpar[jm] += wasserstein_distance(buf_l_mc[:,k], buf_l[:,k])/buf_l.shape[1]
    return KL, KL_dpar

############## data collection
##############
def gather_results(folder, par):
    files = extract_files_from_subfolders(folder, '.dat')
    par_vals = []
    res_ges = []
    res_mc_ges = []
    for file in files:
        with open(file, 'rb') as f:
            fname = os.path.basename(os.path.normpath(file))
            if fname[:3] == 'mc_':
                res_mc_ges.append(pickle.load(f))
            else:
                res_ges.append(pickle.load(f))
                par_vals.append(res_ges[-1].pars[par])
    print('res:', len(res_ges))
    print('mc:', len(res_mc_ges))
    res_ges = [val for _, val in sorted(zip(par_vals, res_ges))]
    par_vals = [val for _, val in sorted(zip(par_vals, par_vals))]
    if len(res_mc_ges) == len(res_ges):
        res_mc_ges = [val for _, val in sorted(zip(par_vals, res_mc_ges))]
    return res_ges, par_vals, res_mc_ges

def extract_files_from_subfolders(root_folder: str, file_extension: str):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    files = []
    for subfolder in subfolders:
        files += glob.glob(os.path.join(subfolder, f'*{file_extension}'))
    return files


def get_standard_metrics(res_ges):
    #metric_names = ['RMSE (train)', 'RMSE (test)', 'logL/N (train)', 'logL/N (test)', 'PDE (mean)', 'PDE (std)']
    #if res_ges[0].pars['learn_rhs'] == 0:
    #    metric_names += [r'$|\lambda - \hat \lambda|$', r'logL $\lambda$']
    #metric_names = ['RMSE (test)', 'logL/N (test)', r'logL $\lambda$', 'W (f)', 'W $(\lambda)$' ]
    metric_names = ['RMSE (true)', 'logL/N (test)', 'W (f)', 'W $(\lambda)$' ]
    #if res_ges[0].pars['learn_rhs'] == 0:
    #    metric_names += [r'logL $\lambda$']
    metric_values, digits, minmax = get_metric_values(res_ges, metric_names)
    
    return metric_names, metric_values, digits, minmax


def get_metric_values(res_ges, metric_names):
    if type(res_ges) is not list:
        res_ges = [res_ges]
    metric_values = []
    digits = []
    minmax = []
    for metric_name in metric_names:
        buf = []
        for jr, res in enumerate(res_ges):
            if metric_name == 'RMSE (train)':            
                buf.append(res.RMSE_ensemble_gesges[:,:,0])
                if jr == 0:
                    digits.append(2)
                    minmax.append(0)
            if metric_name == 'RMSE (test)':            
                buf.append(res.RMSE_ensemble_gesges[:,:,1])
                if jr == 0:
                    digits.append(2)
                    minmax.append(0)
            if metric_name == 'RMSE (true)':            
                buf.append(res.RMSE_ensemble_gesges[:,:,2])
                if jr == 0:
                    if res.pars['x_opt'] == 5:
                        digit = 3
                    elif res.pars['x_opt'] == 102:
                        digit = 3
                    else:
                        digit = 2
                    digits.append(digit)
                    minmax.append(0)
            if metric_name == 'PDE (mean)':            
                buf.append(res.rmse_pde_gesges[:,1,:,:,-1].mean(1))
                if jr == 0:
                    digits.append(3)
                    minmax.append(0)
            if metric_name == 'PDE (std)':            
                buf.append(res.rmse_pde_gesges[:,1,:,:,-1].std(1))
                if jr == 0:
                    digits.append(3)
                    minmax.append(0)
            if metric_name == 'logL/N (train)':            
                buf.append(res.logL_ensemble_gesges[:,:,0]/(res.pars['N_train'] + res.pars['N_train_rhs']))
                if jr == 0:
                    digits.append(1)
                    minmax.append(1)
            if metric_name == 'logL/N (test)':
                buf.append(res.logL_ensemble_gesges[:,:,1]/(res.pars['N_test'] + res.pars['N_test_rhs']))            
                if jr == 0:
                    digits.append(1)
                    minmax.append(1)
            if metric_name == 'logL/N (true)':
                buf.append(res.logL_ensemble_gesges[:,:,2]/(res.pars['N_test'] + res.pars['N_test_rhs']))            
                if jr == 0:
                    digits.append(1)
                    minmax.append(1)
            if metric_name == r'$|\lambda - \hat \lambda|$':
                buf.append(res.ABS_dpar_ensemble_gesges[:,:])
                if jr == 0:
                    if res.pars['x_opt'] == 7:
                        digit = 2 
                    else:
                        digit = 3
                    digits.append(digit)
                    minmax.append(0)
            if metric_name == r'logL $\lambda$':
                buf.append(res.logL_dpar_ensemble_gesges[:,:])
                if jr == 0:
                    if res.pars['x_opt'] == 5:
                        digit = 1
                    elif res.pars['x_opt'] == 102:
                        digit = 1
                    else:
                        digit = 2
                    digits.append(digit)
                    minmax.append(1)  
            if metric_name == r'KL (f)':
                buf.append(res.KL_gesges[:,:])
                if jr == 0:
                    digits.append(2)
                    minmax.append(0)
            if metric_name == r'KL $(\lambda)$':
                buf.append(res.KL_dpar_gesges[:,:])
                print('krak!')
                if jr == 0:
                    digits.append(3)
                    minmax.append(0) 
            if metric_name == r'W (f)':
                buf.append(res.W_gesges[:,:])
                if jr == 0:
                    if res.pars['x_opt'] == 5:
                        digit = 3
                    else:
                        digit = 1
                    digits.append(digit)
                    minmax.append(0)
            if metric_name == r'W $(\lambda)$':
                buf.append(res.W_dpar_gesges[:,:])
                print('krak!')
                if jr == 0:
                    digits.append(3)
                    minmax.append(0)
                    
        metric_values.append(np.array(buf))
    print(np.array(metric_values).shape)
    return np.array(metric_values), digits, minmax


def get_par_captions(par, par_vals):
    if par == 'N_ensemble':
        mc = 'N'
    model_captions = [mc+'='+str(j) for j in par_vals]
    return model_captions


#TODO: move plots to plot .py
def plot_model_results_vs_par(res_ges, res_mc_ges, model_captions, jm, jN, dist_opt = 'mean', ppath='../plots/ensembles/'):
    #from .plots import axplot_ens_logL_vs_dpar, axplot_ens_fx, axplot_ens_fdist, axplot_ens_fdist_mc
    Nr = len(res_ges)
    fig, axs = plt.subplots(3, Nr, figsize=(5*Nr, 4.5*3), squeeze=False)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.1)
    for jr in range(Nr):
        ax2 = axplot_ens_logL_vs_dpar(axs[0,jr], res_ges[jr], jm, jN)
        axplot_ens_fx(axs[1,jr], res_ges[jr], jm, jN)
        if len(res_mc_ges) > 0:
            res_mc = res_mc_ges[0] if len(res_mc_ges) == 1 else res_mc_ges[jr]
            axplot_ens_fdist_mc(axs[2,jr], res_ges[jr], jm, jN, res_mc, dist_opt = dist_opt)
        else:
            axplot_ens_fdist(axs[2,jr], res_ges[jr], jm, jN, dist_opt = dist_opt)
        axs[0,jr].set_title(model_captions[jr])        
        axs[0,jr].set_xlim([0,0.6])
        axs[0,jr].set_ylim([-20,0.])
        ax2.set_xlim([0.1,0.6])
        ax2.set_ylim([-650,0.])

    plt.savefig(ppath+'N_comparison_jm'+str(jm)+'.pdf', bbox_inches='tight')
    
    
def plot_parameter_comparison(res_ges, par, par_vals, metric_names, metric_values, ppath='../plots/ensembles/'):
    from .plots import axplot_stat
    Nm = len(metric_names)
    fig, axs = plt.subplots(1,Nm, figsize=(6*Nm*0.75,4*0.75), squeeze=False)
    for j_metric in range(Nm):
        ax = axs[0,j_metric]
        buf = metric_values[:,:,:,:]
        #mvec = res_ges[0].pars['model_vec'][:] if j_metric not in [1] else res_ges[0].pars['model_vec'][1:-1]
        mvec = res_ges[0].pars['model_vec'][:] if metric_names[j_metric] not in ['logL/N (test)'] else res_ges[0].pars['model_vec'][1:]
        axplot_stat(ax, buf[j_metric], np.log10(par_vals), metric_names[j_metric], par, mvec)
        #axplot_stat(axs[0,j_metric], metric_values[j_metric], par_vals, metric_names[j_metric], par, res.pars['model_vec'])
        #if j_metric == 0:
        #    ax.legend()
        xlabel = r'$N_e$' if par == 'N_ensemble' else par
        ax.set_xlabel('log10 ' + xlabel)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, -0.3), ncol=6, frameon=False)

    plt.savefig(ppath+'N_comparison_eval.pdf', bbox_inches='tight')