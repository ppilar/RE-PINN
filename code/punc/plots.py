# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sbs

from .utils import *
from .utils_eval_results import *
from .axplots import *
    
matplotlib.rcParams.update({'font.size': 14})



def get_t_plot(tmin, tmax, dim):
    if dim == 1:
        t_plot = torch.linspace(tmin[0],tmax[0],200)
        t_plot_shape = [200]
    if dim == 2:
        N = 2500
        sN = int(np.sqrt(N))
        
        t_plot = torch.zeros((N, 2))
        t1 = torch.linspace(tmin[0], tmax[0], sN)
        t2 = torch.linspace(tmin[1], tmax[1], sN)
        buf1, buf2 = torch.meshgrid(t1, t2)
        t_plot[:,0], t_plot[:,1] = buf1.flatten(), buf2.flatten()
        t_plot_shape = [sN, sN]
    return t_plot, t_plot_shape



    
    
def get_yscale(yscale_opt, ax = -1, jax = 0):
    # get correct axis scale
    
    if type(ax) == int:
        if yscale_opt == 0:
            fplt = plt.plot
        else:
            fplt = plt.semilogy
    else:
        if yscale_opt == 0:
            fplt = ax[jax].plot
        else:
            fplt = ax[jax].semilogy
    return fplt
      
    
                    




def plot_residuals(t_train, t_test, f_test, f_net_test, res_test, fres_test, tshape, ppath = -1):
    #tvec = ['true', 'PINN', 'residuals', 'PDE residuals']
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axplot_residuals(axs, t_train, t_test, f_test, f_net_test, res_test, fres_test, tshape)
    if type(ppath) is not int:
        plt.savefig(ppath, bbox_inches='tight')
    plt.show()
    return fig




def plot_ensemble(res, jm, jN, ppath=''):
    fig, axs = plt.subplots(2,4, figsize=(24,9))
    fig.subplots_adjust(hspace=0.3)
    
    if res.pars['dim'] == 1:
        axplot_ens_fx(axs[0,0], res, jm, jN)
        axplot_ens_fdist(axs[0,1], res, jm, jN)
        if res.pars['learn_rhs'] == True:
            axplot_ens_fx(axs[0,2], res, jm, jN, 'rhs')
            axplot_ens_fdist(axs[0,3], res, jm, jN, 'rhs')
    if res.pars['dim'] == 2:
        axplot_ens_mean_2d(axs[0,0], res, jm, jN)
        axplot_true_solution_2d(axs[0,1], res, jm, jN)
        axplot_cuts(axs[0,2], axs[0,3], res, jm, jN)
    axplot_ens_dpar_trajectories(axs[1,0], res, jm, jN)
    axplot_ens_dpar_dist(axs[1,1], res, jm, jN)
    axplot_ens_dpar_hist(axs[1,2], res, jm, jN)
    axplot_ens_logL_vs_dpar(axs[1,3], res, jm, jN)
    
    
    fig.savefig('ensemble.pdf')    
    fig.savefig(ppath + 'x' + str(res.pars['x_opt']) + '_m'+str(jm) + '_N'+str(res.pars['N_ensemble']) + '.pdf', bbox_inches='tight')
    fig.savefig('../plots/ensembles/x' + str(res.pars['x_opt']) +  '_m' + str(jm) + '_N'+str(res.pars['N_ensemble']) + '.pdf', bbox_inches='tight')
    plt.show()
    
    return fig
    
    
def plot_ensemble_comparison(res, jN, ppath='', show_stats=True):     
    matplotlib.rcParams.update({'font.size': 14})
    
    if res.pars['learn_rhs'] == False:
        Nm = len(res.pars['model_vec'])
        
        fig, axs = plt.subplots(Nm, 4, figsize=(24, 4.5*Nm), squeeze=False)
        fig.subplots_adjust(hspace=0.3)
        
        for j, jm in enumerate(res.pars['model_vec']):            
            
            if res.pars['dim'] == 1:
                axplot_ens_fx(axs[j,0], res, jm, jN)
                axplot_ens_fdist(axs[j,1], res, jm, jN)
            if res.pars['dim'] == 2:
                axplot_ens_mean_2d(axs[j,0], res, jm, jN)
                axplot_cuts(None, axs[j,1], res, jm, jN)
            axy2 = axplot_ens_logL_vs_dpar(axs[j,3], res, jm, jN)
            axplot_ens_dpar_hist(axs[j,2], res, jm, jN)
            
            axs[j,0].set_ylabel(get_mname(jm, rep_space = get_rep_space(res.pars)))

                
            
        fig.savefig(ppath + 'ensemble_comparison_jN'+str(jN)+'.pdf', bbox_inches='tight')    
        plt.show()
        
    else:
        Nm = len(res.pars['model_vec'])
        fsx = 30 if show_stats else 24
        Np = 5 if show_stats else 4
        fig, axs = plt.subplots(Nm, Np, figsize=(fsx, 4.5*Nm), squeeze=False)
        fig.subplots_adjust(hspace=0.3)
        
        for j, jm in enumerate(res.pars['model_vec']):
            if res.pars['dim'] == 1:
                axplot_ens_fx(axs[j,0], res, jm, jN)
                axplot_ens_fdist(axs[j,1], res, jm, jN)
            if res.pars['dim'] == 2:
                axplot_ens_mean_2d(axs[j,0], res, jm, jN)
                axplot_ens_std_2d(axs[j,1], res, jm, jN)
            
            axplot_ens_fx(axs[j,2], res, jm, jN, 'rhs')
            axplot_ens_fdist(axs[j,3], res, jm, jN, 'rhs')
            axs[j,0].set_ylabel(get_mname(jm, rep_space = get_rep_space(res.pars)))
            
            if show_stats:
                axplot_statistics(axs[j,4], res, jm, jN)
            
        fig.savefig(ppath + 'ensemble_comparison_jN'+str(jN)+'.pdf', bbox_inches='tight')    
        plt.show()
        
        
def pairwise_plot(pars, data, feature_names=None, bins=10, mc_samples=None, jm_vec = None, opt='', plot_opt='', ppath = '../plots/'):
    import seaborn as sbs
    
    plot_mc = False if mc_samples is None else True
    
    if data.ndim == 2:
        data = np.expand_dims(data,0)        
    _, n_samples, n_features = data.shape   
    n_models = pars['Nmodel']


    figsize = (5.5*n_features, 5*n_features)
    if feature_names is None:
        #
        if pars['x_opt'] == 1:
            feature_names = [r"$f_0$", r"$\lambda$"]
        elif pars['x_opt'] == 7:
            feature_names = [r"$f_0$", r"$f_1$", r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]
        else:
            feature_names = [r"$\lambda_{%i}$"%(i) for i in range(n_features)]
    fig, axes = plt.subplots(n_features, n_features, figsize=figsize, squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if jm_vec is not None:
        jm_vec = jm_vec.copy()
    if opt == 'only_mc':
        jm_vec = [0]
    elif jm_vec == None:
        jm_vec = [m for m in pars['model_vec']]
        if plot_mc:
            jm_vec += [pars['Nmodel']+1]
    else:
        if plot_mc:
            jm_vec += [pars['Nmodel']+1]
    
    
    for jm in jm_vec:
        data_buf = data[jm] if jm <= pars['Nmodel'] else mc_samples
        alpha = 0.75 if jm <= pars['Nmodel'] else min(25/mc_samples.shape[0], 1)
        color = 'C'+str(jm) if jm <= pars['Nmodel'] else 'dimgrey'
            
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
    
                if i == j:
                    #ax.hist(data[jm, :, i], bins=bins, color=hist_color, edgecolor='black')
                    label = get_label(jm) if i == 0 else None
                    if plot_opt == 'sbs' or jm == pars['Nmodel'] + 1 or pars['N_ensemble'] > 40:
                        sbs.kdeplot(data_buf[:, i], ax=ax, color=color, label=get_label(jm))
                        if jm == pars['Nmodel'] + 1:
                            if pars['x_opt'] == 1:
                                if i == 0:
                                    ax.set_ylim([0,3])
                                if i == 1:
                                    ax.set_ylim([0,25])
                                    
                    else:
                        counts, _, _ = ax.hist(data_buf[:, i], bins=bins, histtype='step', linewidth=1.5, color=color, label=get_label(jm), density=True)
                        if jm == pars['Nmodel'] + 1:
                            ax.set_ylim([0,5*counts.max()])
                else:
                    
                    if plot_opt == 'sbs' or jm == pars['Nmodel'] + 1: #[0.2, 0.5, 0.8] #[0.683, 0.955, 0.997]
                        sbs.kdeplot(x=data_buf[:, j], y=data_buf[:, i], ax=ax, levels=[0.003, 0.045, 0.317], color=color)
                    else:
                        ax.scatter(data_buf[:, j], data_buf[:, i], alpha=alpha, color=color, s=15)
    
                # Axis labels
                if i == n_features - 1:
                    ax.set_xlabel(feature_names[j], fontsize=20)
    
                if j == 0:
                    ax.set_ylabel(feature_names[i], fontsize=20)
                
                    
                

    axes[0,0].legend()
    plt.tight_layout()
    
    plt.savefig(ppath + 'lambda_correlations.pdf', bbox_inches='tight')
    plt.show()
    return fig
    
    

def print_ensemble_comparison_table(res, jN, fpath='', fname='stats', opt=''):  
    #create latex table entries of statistics 
    import sys
    
    #metric_names = ['RMSE (train)', 'RMSE (test)', 'RMSE (true)', 'logL/N (train)', 'logL/N (test)', 'logL/N (true)']
    metric_names = ['RMSE (true)', 'logL/N (test)']
    if res.pars['learn_rhs'] == 0:
        metric_names += [r'$|\lambda - \hat \lambda|$', r'logL $\lambda$']
    if hasattr(res, 'W_gesges'):
        metric_names += ['W (f)']
    if hasattr(res, 'W_dpar_gesges'):  
        metric_names += [r'W $(\lambda)$']
    model_captions = [get_label(jm, rep_space=get_rep_space(res.pars), opt='table', opt2=opt) for jm in res.pars['model_vec']]
    
    metric_values, digits, minmax = get_metric_values(res, metric_names)
    metric_values = metric_values.squeeze(1)
    if opt == '':
        print_comparison_table(metric_names, metric_values[:,res.pars['model_vec'],:jN+1], model_captions, digits, minmax, fpath=fpath, fname=fname)
    elif opt == 'flipped':
        print_comparison_table_flipped(metric_names, metric_values[:,res.pars['model_vec'],:jN+1], model_captions, digits, minmax, fpath=fpath, fname=fname)
    
    
def print_comparison_table(metric_names, metric_values, model_captions, digits, minmax, fpath='', fname='stats'):
    import sys
    
    Nmetric = len(metric_names)
    Nmodel = len(model_captions)
    
    factors = [1]*Nmetric
    
    model_str = '& '
    for jm, caption in enumerate(model_captions):
        model_str += caption
        if jm == Nmodel-1:
            model_str += r' \\ '
        else:
            model_str += r' & '
    
    cstr = 'c '
    for jc in range(Nmodel):
        cstr += 'c '
    
    with open(fpath+fname+'.txt', 'w') as f:
        original_stdout = sys.stdout 
        sys.stdout = f # Change the standard output to the file we created.
        
        print('{' + cstr + '}')
        print(r'\hline')
        print(model_str) # print model names    
        print(r'\cline{2-'+str(1+Nmodel)+'}')
        for j in range(Nmetric):   
            factor = factors[j]
            dig = digits[j]
            buf = metric_values[j]
            crit = np.argmin if minmax[j] == 0 else np.argmax
            jm_best = crit(buf.mean(1))

            
            print( metric_names[j], end='')
            for jm in range(Nmodel):
                if jm == jm_best:
                    print(r' & \textbf{' + f"{np.round(buf[jm,:].mean()/factor, dig):.{dig}f}" + r'}$\pm$' + f"{np.round(buf[jm,:].std()/factor, dig):.{dig}f}", end='') 
                else:
                    print(r' & ' + f"{np.round(buf[jm,:].mean()/factor, dig):.{dig}f}" + r'$\pm$' + f"{np.round(buf[jm,:].std()/factor, dig):.{dig}f}", end='')                   
                    
            print(r'\\') 
        print(r'\hline')
        
        sys.stdout = original_stdout # Reset the standard output to its original value

def digits_before_decimal(x):
    return len(str(int(abs(x))))
        
def print_comparison_table_flipped(metric_names, metric_values, model_captions, digits, minmax, fpath='', fname='stats'):
    import sys
    
    Nmetric = len(metric_names)
    Nmodel = len(model_captions)
    
    factors = [1]*Nmetric
    
    metric_str = '& '
    for jm, caption in enumerate(metric_names):
        metric_str += caption
        if jm == Nmetric - 1:
            metric_str += r' \\ '
        else:
            metric_str += r' & '
    
    
    cstr = 'l '
    for jc in range(Nmetric):
        cstr += 'c '

    with open(fpath+fname+'_flipped.txt', 'w') as f:
        original_stdout = sys.stdout 
        sys.stdout = f # Change the standard output to the file we created.
        
        print('{' + cstr + '}')
        print(r'\hline') 
        print(metric_str)   
        print(r'\cline{2-'+str(1+Nmetric)+'}')
        for jm in range(Nmodel): 
            print(model_captions[jm], end='')
            for j in range(Nmetric): #jm_vec: #loop over models
                factor = factors[j]
                dig = digits[j]
                buf = metric_values[j]
                crit = np.argmin if minmax[j] == 0 else np.argmax
                jm_best = crit(buf.mean(1))  
                sbuf = '-' if buf[jm,:].mean() < 0 else ''
                dbuf = digits_before_decimal(buf[jm,:].mean()) - digits_before_decimal(buf[jm,:].std())
                d0 = '' if dbuf > 0 else ''.join('0' for j in range(dbuf))
                d1 = '' if dbuf < 0 else ''.join('0' for j in range(dbuf))
                
            
                if jm == jm_best:
                    print(r' & \textbf{' + r'\phantom{'+d0+'}' + f"{np.round(buf[jm,:].mean()/factor, dig):.{dig}f}" + r'}$\pm$' + f"{np.round(buf[jm,:].std()/factor, dig):.{dig}f}"+r'\phantom{'+sbuf+'}' + r'\phantom{'+d1+'}', end='') 
                else:
                    print(r' & ' + r'\phantom{'+d0+'}' + f"{np.round(buf[jm,:].mean()/factor, dig):.{dig}f}" + r'$\pm$' + f"{np.round(buf[jm,:].std()/factor, dig):.{dig}f}"+r'\phantom{'+sbuf+'}' + r'\phantom{'+d1+'}', end='')                   
                    
            print(r'\\') 
        print(r'\hline')
        
        sys.stdout = original_stdout # Reset the standard output to its original value
        
        
        
        
#%% paper plots

def plot_distribution_and_samples(res, res_mc, jN, jm_vec, plist, ppath, pname):
    Nm = len(jm_vec)
    dist_opt = 'median'
    
    Np = len(plist)
    fig, axs = plt.subplots(Np, Nm, figsize=(Nm*5*0.7, (0.7*4.5 + 0.3)*Np), squeeze=False)
    fig.subplots_adjust(wspace=0.)
    fig.subplots_adjust(hspace=0.3)
    for j, jm in enumerate(jm_vec):
        print(jm)
        for jp, p in enumerate(plist):
            if p == 'f':
                axplot_ens_fdist_mc(axs[jp,j], res, jm, jN, res_mc, dist_opt=dist_opt)
            if p == 's':
                axplot_ens_fx(axs[jp,j], res, jm, jN)
                if jp == 0:
                    axs[jp,j].set_title(get_label(jm))
                else:
                    axs[jp,j].set_title('')
            if p[0] == 'l':
                jl = int(p[1:])
                axplot_ens_ldist_mc(axs[jp,j], res, jm, jN, jl=jl, res_mc = res_mc, label='ensemble')
            if j > 0:
                axs[jp,j].get_legend().remove()
                axs[jp,j].get_yaxis().set_visible(False)
            
            if res.pars['x_opt'] == 102:
                axs[jp,j].get_xticklabels()[-1].set_visible(False)

                
    fig.savefig(ppath+pname+'.pdf', bbox_inches='tight')
    
    
def plot_overview(res, res_mc, jm, jN, ppath, pname):
    dist_opt = 'median'
    fig, axs = plt.subplots(1, 3, figsize=(3*6.5*0.8, 4.5*0.8))
    fig.subplots_adjust(wspace=0.16)
    axplot_ens_fdist_mc(axs[0], res, jm, jN, res_mc, dist_opt=dist_opt)
    axplot_f_fdist(axs[0], res, 0, jN, dist_opt='median', colors = ['C1'], quantiles=[0.05, 0.25], label=get_label(0))
    axs[0].legend()
    for jl in range(res.pars['Npar']):
        axplot_lambda_kde(axs[1], res, 0, jN, jl=jl, color='C1', label=get_label(0))
        axplot_ens_ldist_mc(axs[1], res, jm, jN, jl=jl, res_mc = res_mc, label='ensemble')
    handles, labels = axs[1].get_legend_handles_labels()
    handles = handles[:-4]
    labels = labels[:-4]
    axs[1].legend(handles, labels, loc='upper center')
    axs[1].set_ylabel('')
    axs[1].set_xlabel('')
    axplot_ens_fx(axs[2], res, jm, jN)
    if res.pars['x_opt'] == 5:
        axs[1].set_xlim([-0.1,1.3])
        axs[1].set_title(r'$p(\gamma|\mathcal{D})$                     $p(\omega|\mathcal{D})$  ')
        axs[0].set_ylim([-0.75, 1.2])
        axs[2].set_ylim([-0.75, 1.2])
    else:
        axs[1].set_title(r'$p(\lambda|\mathcal{D})$')
    fig.savefig(ppath+pname+'.pdf', bbox_inches='tight')
    
def plot_overview_VI(res, res_mc, jm, jN, ppath, pname):
    dist_opt = 'median'
    fig, axs = plt.subplots(1, 3, figsize=(3*6.5*0.8, 4.5*0.8))
    fig.subplots_adjust(wspace=0.16)
    axplot_ens_fdist_mc(axs[0], res, jm, jN, res_mc, dist_opt=dist_opt)
    axs[0].legend()
    for jl in range(res.pars['Npar']):
        axplot_ens_ldist_mc(axs[1], res, jm, jN, jl=jl, res_mc = res_mc, label='VI')
    handles, labels = axs[1].get_legend_handles_labels()
    handles = handles[:-3]
    labels = labels[:-3]
    axs[1].legend(handles, labels, loc='upper center')
    axs[1].set_ylabel('')
    axs[1].set_xlabel('')
    axplot_ens_fx(axs[2], res, jm, jN)
    if res.pars['x_opt'] == 5:
        axs[1].set_xlim([-0.1,1.3])
        axs[1].set_title(r'$p(\gamma|\mathcal{D})$                     $p(\omega|\mathcal{D})$  ')
        axs[0].set_ylim([-0.75, 1.2])
        axs[2].set_ylim([-0.75, 1.2])
    else:
        axs[1].set_title(r'$p(\lambda|\mathcal{D})$')
    
    axs[0].set_title('variational inference')
    axs[2].set_title('samples')
    
    fig.savefig(ppath+pname+'.pdf', bbox_inches='tight')
    
    
def plot_frontpage_ds1(res, res_mc, jm, jN, ppath, pname):
    #from punc.plots import axplot_f_mc
    
    dist_opt = 'median'
    fig, ax = plt.subplots(1,1, figsize=(8*0.6,8*0.6))
    axplot_ens_fdist(ax, res, 5, 0, None, dist_opt=dist_opt, quantiles = [0.05, 0.25]) 
    axplot_f_mc(ax, res, 5, 0, res_mc, dist_opt=dist_opt, quantiles = [0.05, 0.25])     
    axplot_ens_fdist(ax, res, 0, 0, None, dist_opt=dist_opt, colors=['darkorange'], quantiles = [0.05, 0.25]) 
    ax.set_title('')
    
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = get_label(5)
    labels[3] = get_label(0)
    labels[2] = get_label(-1)#'Monte Carlo'
    ivec = [0,3,2]
    handles = [handles[i] for i in ivec]
    labels = [labels[i] for i in ivec]
    ax.legend(handles, labels)    
    
    lines = ax.get_lines()
    lines[1].remove() 
    lines[8].remove() 
    
    ax.collections[3].remove()
    ax.collections[6].remove()
    ax.collections[4].remove()
    ax.set_xlim([0,13])
    ax.set_ylim([0,30])
    
    fig.savefig(ppath+pname+'.pdf', bbox_inches='tight')
    
    
def plot_data_ds102(res, jm, jN, ppath, pname):
    res.pars['real_data'] = 0
    fig, ax = plt.subplots(1,1, figsize=(8,10))
    #axplot_ens_mean_2d(ax, res, jm, jN)
    axplot_true_solution_2d(ax, res, jm, jN)
    ax.scatter(res.data[jN]['t_train'][:,0], res.data[jN]['t_train'][:,1], color='k', marker='x', s=10)
    tcut, (cut0, cut1), t_test, (y_test0, y_test1) = get_cuts(res, jm, jN, [0,25])
    ax.scatter(0.025*np.ones(t_test.shape[0]), t_test, color='r', marker='x', s=10)
    ax.scatter(1*np.ones(t_test.shape[0]), t_test, color='r', marker='x', s=10)
    ax.set_xlim([0,1.975])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('')
    fig.savefig(ppath + pname +'.pdf', bbox_inches='tight')
    
def plot_samples_ds102(res, jm_vec, jN, ppath, pname, Ns=4):
    vmin, vmax = [-.5, 1.]
    Nm = len(jm_vec)
    
    fig, axs = plt.subplots(Nm, Ns, figsize=(4*Ns+0.2,4*Nm))
    fig.subplots_adjust(wspace=0.)
    fig.subplots_adjust(hspace=0.1)
    
    for j, jm in enumerate(jm_vec):
        for k in range(Ns):            
            axplot_refined_imshow(axs[j, k], res.xnp_gesges[jm,k,jN,:], res.pars['t_plot_shape'], res.data[jN]['t_plot'], vmin=vmin, vmax=vmax)
            if k > 0:
                axs[j,k].get_yaxis().set_visible(False)
            else:
                axs[j,k].set_ylabel(get_label(jm)+'\n x')
            if jm != jm_vec[-1]:
                axs[j,k].get_xaxis().set_visible(False)
            else:
                axs[j,k].set_xlabel('t')

    fig.savefig(ppath + pname + '.pdf', bbox_inches='tight')