# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sbs

from .utils import *
from .utils_eval_results import *


matplotlib.rcParams.update({'font.size': 14})


def axplot_fill_between(ax, arr, xvec = -1, color=-1, label=''):
    if type(xvec) == int:
        xvec = np.range(arr.shape[1])
    if type(color) == int:
        color = 'b'
    
    mu = arr.mean(0)
    #mu = np.median(arr, axis=0)
    std = arr.std(0)
    
    ax.plot(xvec, mu, color=color, label=label)
    ax.fill_between(xvec, mu+std, mu-std, color=color, alpha=0.5)
    

def axplot_stat(ax, stat, pvec, sname, pname, jmodel_vec = [3,2,0,1]):
    mu = np.mean(stat[:,:,:],-1)
    std = np.std(stat[:,:,:],-1)
    stat = np.swapaxes(stat, 0, 1)
    stat = np.swapaxes(stat, 1, 2)
    axplot_model_comp(ax, jmodel_vec, stat, pvec)
    ax.set_title(sname)
    ax.set_xlabel(pname)

def axplot_model_comp(ax, jmodel_vec, arr, xvec):
    for jm in jmodel_vec:
        l = get_label(jm)
        c = get_color(jm)
        axplot_fill_between(ax,arr[jm],xvec, color=c, label=l)
    #ax.legend()
    
    
def get_ylim(x_opt, s_opt):
    ylim = -1
    if s_opt == 'dpar':
        if x_opt == 1:
            ylim=[0.1,0.4]
            
    if s_opt == 'rmse':
        if x_opt == 1:
            ylim=[0,15]
        if x_opt == 101:
            ylim=[0,0.4]
            
    if s_opt == 'logL val':
        if x_opt == 1:
            ylim=[-15,0]
        if x_opt == 101:
            ylim = [-1,0.2]
            
    if s_opt == 'NLL val':
        if x_opt == 1:
            ylim=[0,15]
        if x_opt == 101:
            ylim = [-0.2,1]
            
    if s_opt == 'dpde':
        if x_opt == 1:
            ylim=[0,1]
        if x_opt == 3:
            ylim=[0,0.1]
        if x_opt == 101:
            ylim=[0,0.01]
    
    return ylim
    
def axplot_statistics(ax, stat_gesges, xvec = -1, model_vec = [3,2,0,1], title='', clabel=True, xlabel='iterations', x_opt = -1, s_opt='', step=1, legend=True):
    Nmodel, Nrun, Nx = stat_gesges.shape
    if type(xvec) == int:
        xvec = np.arange(Nx)*step
    
    for jm in model_vec:
        label = get_label(jm) if clabel else ''
        buf = stat_gesges[jm,:,:]
        axplot_fill_between(ax, buf, xvec = xvec, color=get_color(jm), label=label)
    
    ylim = get_ylim(x_opt, s_opt)
    if type(ylim) != int: ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if legend: ax.legend()
   

def axplot_dpar_statistics(ax, dpar_gesges, dpar_true, title='', x_opt=0, step=1, model_vec = [3,2,0,1]):
    xvec = np.arange(dpar_gesges.shape[-1])
    for jp in range(dpar_gesges.shape[1]):
        clabel = True if jp == 0 else False
        axplot_statistics(ax, dpar_gesges[:,jp,:,:], clabel=clabel, title=title, x_opt = x_opt, s_opt='dpar', step=step, model_vec = model_vec)
        ax.hlines(dpar_true[jp],0,xvec[-1]*step,color='gray',linestyle='--')
        
        

        
def axplot_logL_comparison(ax, logLG_gesges, logLebm_gesges, x_opt, model_vec = [3,2,0,1], step=1):
    Nmodel, _, Nrun, Nx = logLebm_gesges.shape
    buf = np.zeros([Nmodel, Nrun, Nx])
    buf = logLebm_gesges[:,1,:,:]
    buf[0,:,:] = logLG_gesges[0,1,:,:]
    buf[2,:,:] = logLG_gesges[2,1,:,:]   
    axplot_statistics(ax, -buf, title='NLL validation', x_opt = x_opt, model_vec = model_vec, s_opt = 'NLL val', step=step)



def axplot_residuals(axs, t_train, t_test, f_test, f_net_test, res_test, fres_test, tshape, x_opt=-1):
    tvec = ['true', 'PINN', 'residuals', 'PDE residuals']
    for j in range(4):
        ax = axs[j]
        if j == 0:
            buf = f_test
        if j == 1:
            buf = f_net_test
        if j == 2:
            buf = torch.log10(torch.abs(res_test))
        if j == 3:
            buf = torch.log10(torch.abs(fres_test))
            
        extent = [t_test[:,0].min().item(), t_test[:,0].max().item(), t_test[:,1].min().item(), t_test[:,1].max().item()]
        if j < 2:
            if x_opt == 101:
                vmin = -0.5
                vmax = 1.4
            else:
                vmin = -0.2
                vmax = 1
        else:
            vmin = -8
            vmax = 0
            
        if x_opt == 101:
            ax.imshow(buf.reshape(tshape).detach().cpu().numpy(), extent=extent, vmin=vmin, vmax=vmax)
            ax.set_xlabel('y')
        else:
            ax.imshow(np.flip(buf.reshape(tshape).T.detach().cpu().numpy(),0), extent=extent, aspect=2, vmin=vmin, vmax=vmax)
            if type(t_train) is not int:
                ax.scatter(t_train[:,0].cpu(), t_train[:,1].cpu(), color='r', marker='x', s=2)
            ax.set_xlabel('t')
            ax.set_xlim(-0.01,2)
            
        if j == 0:
            ax.set_ylabel('x')
        ax.set_title(tvec[j])
        
        
def axplot_distribution(ax, arr, t_plot=-1, color='#1f77b4', label='', dist_opt='mean', plot_opt='', quantiles = [0.05, 0.25]):
    if type(t_plot) == int:
        t_plot = np.arange(arr.shape[-1])
    if t_plot.ndim > 1:
        t_plot = t_plot.squeeze()
        
    line_style = '--' if plot_opt == 'dashed' else '-'
    if label == '': label = dist_opt
        
    if dist_opt == 'mean':
        fmean = arr.mean(0)
        fstd = arr.std(0)
        ax.plot(t_plot, fmean, linestyle=line_style, label=label, zorder=9, color=color)   
        
        if plot_opt == 'dashed':
            ax.plot(t_plot, fmean + 3*fstd, linestyle='--', color=color)
            ax.plot(t_plot, fmean - 3*fstd, linestyle='--', color=color)
        else:
            ax.fill_between(t_plot, fmean + fstd, fmean - fstd, alpha=0.5, color=color)
            ax.fill_between(t_plot, fmean + 2*fstd, fmean - 2*fstd, alpha=0.4, color=color)
            ax.fill_between(t_plot, fmean + 3*fstd, fmean - 3*fstd, alpha=0.3, color=color)
    
    if dist_opt == 'median':
        #alpha_list = [0.4, 0.3, 0.2]
        ax.plot(t_plot, np.quantile(arr, 0.5, axis=0), linestyle = line_style, label=label, zorder=9, color=color)
        for jq, q in enumerate(quantiles):
            if plot_opt == 'dashed':
                # if jq in [0,1]:
                ax.plot(t_plot, np.quantile(arr, q, axis=0), linestyle='--', color=color)
                ax.plot(t_plot, np.quantile(arr, 1-q, axis=0), linestyle='--', color=color)         
            else:
                ax.fill_between(t_plot, np.quantile(arr, q, axis=0), np.quantile(arr, 1-q, axis=0), alpha=get_alpha(q), color=color)
   
def get_alpha(q):
    if q == 0.25:
        return 0.5
    if q == 0.05:
        return 0.3
                

                
def axplot_logL_vs_nu(ax, pars, dpar_ges, logL_ges, color=None):
    dpars = []
    logLs = []
    for j_ens in range(pars['N_ensemble']):
        dpars.append(dpar_ges[j_ens].item())
        logLs.append(logL_ges[j_ens].item())
        
    
    ax.scatter(dpars, logLs, c=color)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('logL')
    #ax.set_yscale('log')

def axplot_refined_imshow(ax, buf, tshape, t_test, vmin=None, vmax=None):
    extent = [t_test[:,0].min().item(), t_test[:,0].max().item(), t_test[:,1].min().item(), t_test[:,1].max().item()]
    ax.imshow(np.flip(buf.reshape(tshape).T,0), extent=extent, aspect=2, vmin=vmin, vmax=vmax)
    
    
def axplot_true_solution_2d(ax, res, jm, jN):
    axplot_refined_imshow(ax, res.data[jN]['f_test'], res.pars['tshape'], res.data[jN]['t_test'])
    ax.set_title('true solution')

def axplot_ens_mean_2d(ax, res, jm, jN):
    axplot_refined_imshow(ax, res.xnp_gesges[jm,:,jN,:].mean(0), res.pars['t_plot_shape'], res.data[jN]['t_plot'])
    ax.scatter(res.data[jN]['t_train'][:,0], res.data[jN]['t_train'][:,1], color='k', marker='x', s=2)
    
    ax.set_title('ensemble mean')
    ax.set_xlabel('t')
    
    

def axplot_ens_std_2d(ax, res, jm, jN):
    axplot_refined_imshow(ax, res.xnp_gesges[jm,:,jN,:].std(0), res.pars['t_plot_shape'], res.data[jN]['t_plot'])
    ax.scatter(res.data[jN]['t_train'][:,0], res.data[jN]['t_train'][:,1], color='r', marker='x', s=2)
    ax.set_title('ensemble std')
    ax.set_xlabel('t') 

def axplot_ens_fx(ax, res, jm, jN, opt='x'):
    buf = res.data[jN]['f_plot_rhs'] if opt == 'rhs' else res.data[jN]['f_plot']
    if res.pars['real_data'] == 0:
        ax.plot(res.data[jN]['t_plot'], buf, label='true', color='k', zorder=10)
    for j in range(res.pars['N_ensemble']):
        buf = res.xnp_rhs_gesges[jm,j,jN,:] if opt == 'rhs' else res.xnp_gesges[jm,j,jN,:]
        ax.plot(res.data[jN]['t_plot'], buf, color='C'+str(j), lw=0.5)
    if opt == 'rhs':
        ax.scatter(res.data[jN]['t_train_rhs'], res.data[jN]['y_train_rhs'], color='k', s=7, zorder=10)        
    else:
        for k in range(res.pars['output_dim']):
            ax.scatter(res.data[jN]['t_train'], res.data[jN]['y_train'][:,k], color='k', s=7, zorder=10)
    ax.legend()
    tstr = 'ensemble member predictions'
    tstr2 = ''
    ax.set_title(tstr + tstr2)
    ax.set_ylim(res.pars['ylim'])
    ax.set_xlabel('t')
    
    

    
def axplot_statistics(ax, res, jm, jN):
    buf = (r'logL train=%.2f '%(res.logL_ensemble_gesges[jm,jN,0])
            + '\n' + r'logL test=%.2f'%(res.logL_ensemble_gesges[jm,jN,1])
            + '\n' + r'RMSE train=%.2f '%(res.RMSE_ensemble_gesges[jm,jN,0])
            + '\n' + r'RMSE test=%.2f'%(res.RMSE_ensemble_gesges[jm,jN,1])
            + '\n'
            + '\n' + 'N=' + str(jN+1)
            + '\n' + r'logL train=%.2f $\pm$ %.2f '%(res.logL_ensemble_gesges[jm,:jN+1,0].mean(), res.logL_ensemble_gesges[jm,:jN+1,0].std())
            + '\n' + r'logL test=%.2f $\pm$ %.2f'%(res.logL_ensemble_gesges[jm,:jN+1,1].mean(), res.logL_ensemble_gesges[jm,:jN+1,1].std())
            + '\n' + r'RMSE train=%.2f $\pm$ %.2f '%(res.RMSE_ensemble_gesges[jm,:jN+1,0].mean(), res.RMSE_ensemble_gesges[jm,:jN+1,0].std())
            + '\n' + r'RMSE test=%.2f $\pm$ %.2f'%(res.RMSE_ensemble_gesges[jm,:jN+1,1].mean(), res.RMSE_ensemble_gesges[jm,:jN+1,1].std())
            )
        
    ax.text(0.01, 0.99, 
            buf,
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes)
    
    
    
def get_cuts(res, jm, jN, inds, axis='x'):
    if axis == 'x':
        tcut = res.data[jN]['t_test'].reshape(*res.pars['tshape'],res.pars['dim'])[0,:,1]    
    else:
        tcut = res.data[jN]['t_test'].reshape(*res.pars['tshape'],res.pars['dim'])[:,0,0]    
            
    t_test = tcut[::4]
    
    cuts, y_tests = [], []
    for ind in inds:
        if axis == 'x':
            cuts.append(res.xnp_gesges[jm,:,jN,ind,:,0])
            y_tests.append(res.data[jN]['y_test'].reshape(*res.pars['tshape'])[ind,::4])
        else:
            cuts.append(res.xnp_gesges[jm,:,jN,:, ind,0])
            y_tests.append(res.data[jN]['y_test'].reshape(*res.pars['tshape'])[::4, ind])
        
    return tcut, cuts, t_test, y_tests

def axplot_cuts(ax0, ax1, res, jm, jN, dist_opt='mean'):
    tcut, (cut0, cut1), t_test, (y_test0, y_test1) = get_cuts(res, jm, jN, [0,25])
    
    if ax0 is not None:
        for j in range(res.pars['N_ensemble']):
            ax0.plot(tcut, cut0[j], color='C'+str(j))
            ax0.plot(tcut, cut1[j], color='C'+str(j))
    
    ax1.plot(tcut, res.data[jN]['f_test'].reshape(*res.pars['tshape'])[0,:], color='k', label='true')
    ax1.plot(tcut, res.data[jN]['f_test'].reshape(*res.pars['tshape'])[25,:], color='k')
    axplot_distribution(ax1, cut0, tcut, color='C0', dist_opt=dist_opt, label='t=0.0')
    axplot_distribution(ax1, cut1, tcut, color='C2', dist_opt=dist_opt, label='t=1.5')
    
    ax1.scatter(t_test, y_test0, color='r', s=10)
    ax1.scatter(t_test, y_test1, color='r', s=10)
    
    ax1.legend()
    
    
    
def axplot_f_fdist(ax, res, jm, jN, dist_opt='mean', opt='x', label = '', colors = ['C0', 'C2'], quantiles=[0.05, 0.25]):
    buf_xnp = res.xnp_rhs_gesges[jm,:,jN,:] if opt == 'rhs' else res.xnp_gesges[jm,:,jN,:]
    for j in range(buf_xnp.shape[-1]):
        axplot_distribution(ax, buf_xnp[:,:,j], res.data[jN]['t_plot'], label=label, color=colors[j], dist_opt=dist_opt, quantiles=quantiles)
   
    
def axplot_ens_fdist(ax, res, jm, jN, opt='x', dist_opt='mean', title='', colors = ['C0', 'C2'], quantiles=[0.05, 0.25]):
    axplot_f_fdist(ax, res, jm, jN, dist_opt=dist_opt, opt=opt, colors = colors, quantiles=quantiles)
    buf_f_plot = res.data[jN]['f_plot_rhs'] if opt == 'rhs' else res.data[jN]['f_plot']
    if res.pars['real_data'] == 0:
        ax.plot(res.data[jN]['t_plot'], buf_f_plot, label='true', color='k', zorder=10)
    if opt == 'rhs':
        ax.scatter(res.data[jN]['t_train_rhs'], res.data[jN]['y_train_rhs'], color='k', s=7, zorder=10)   
        ax.scatter(res.data[jN]['t_test_rhs'], res.data[jN]['y_test_rhs'], color='r', s=4, zorder=10)       
    else:
        for j in range(res.pars['output_dim']):
            ax.scatter(res.data[jN]['t_train'], res.data[jN]['y_train'][:,j], color='k', s=7, zorder=10) 
            ax.scatter(res.data[jN]['t_test'], res.data[jN]['y_test'][:,j], color='r', s=4, zorder=10)   
    
    if title == '':
        title = 'ensemble distribution'
        title += ' (g)' if opt == 'rhs' else ' (f)'

    ax.set_title(title)
    ax.set_ylim(res.pars['ylim'])
    ax.legend()
    ax.set_xlabel('t')
    


def axplot_lambda_kde(ax, res, jm, jN, jl, color='C0', label=''):
    data_buf = res.dpar_gesges[jm,jl,:,jN,-1]
    if label == '':
        label = get_label(jm)
    sbs.kdeplot(data_buf, ax=ax, color=color, label=label)
    
    
def axplot_ens_ldist_mc(ax, res, jm, jN, jl = 0, res_mc = None, plot_opt='', label=''):
    axplot_lambda_kde(ax, res, jm, jN, jl, color='C0', label=label)
    
    if res_mc is not None:
        data_buf = res_mc.dpar_gesges[0,jl+1,:,jN]
        sbs.kdeplot(data_buf, ax=ax, color='dimgrey', label=get_label(res.pars['Nmodel'] + 1))
        
    if res.pars['x_opt'] == 1:
        ylim = [0,25]
    elif res.pars['x_opt'] == 5:
        ylim = [0,20] if res.pars['t_opt'] == 'gap' else [0,25] 
        if jl == 0:
            xlim = [0.8, 1.2] if res.pars['t_opt'] == 'gap'  else [0.85, 1.3]
        if jl == 1:
            xlim = [0,0.4] if res.pars['t_opt'] == 'gap'  else [-0.01, 0.45]
    elif res.pars['x_opt'] == 7:
        if jl == 0:
            xlim = [0.5,2.5]
            ylim = [0,15]
        elif jl == 1:
            xlim = [0.,1.5]
            ylim = [0,10]
        elif jl == 2:
            xlim = [1.,5.]
            ylim = [0,5]
        elif jl == 3:
            xlim = [0.5,1.5]
            ylim = [0,10]
    elif res.pars['x_opt'] == 102:
        xlim = [0.16, 0.24]
        ylim = [0,100]
    if 'xlim' in locals():
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(r'p('+ get_par_name(res.pars, jl) + r'$|\mathcal{D}$)')
    ax.set_xlabel(get_par_name(res.pars, jl))
    ax.vlines(res.pars['dpar'][jl], ylim[0], ylim[1], color='grey', linestyle='--', label='true')
    ax.legend()
    

def axplot_f_mc(ax, res, jm, jN, res_mc = None, dist_opt='mean', colors=['C0', 'C2'], quantiles=[0.05, 0.25]):
    if res_mc is not None:
        buf_xnp = res_mc.xnp_gesges[0,:,jN,:]       
        axplot_distribution(ax, buf_xnp, res.data[jN]['t_plot'].squeeze(), label = get_label(-1), color='dimgrey', dist_opt=dist_opt, plot_opt = 'dashed', quantiles=quantiles)
    
    
def axplot_ens_fdist_mc(ax, res, jm, jN, res_mc = None, dist_opt='mean', colors=['C0', 'C2'], quantiles=[0.05, 0.25]):
    if res.pars['dim'] == 1:
        axplot_ens_fdist(ax, res, jm, jN, dist_opt=dist_opt, title=get_label(jm), colors=colors, quantiles=quantiles)
        axplot_f_mc(ax, res, jm, jN, res_mc = res_mc, dist_opt=dist_opt, colors=colors, quantiles=[0.05])
        ax.legend()
    else:
        axplot_cuts(None, ax, res, jm, jN, dist_opt=dist_opt)
        ax.set_title(get_label(jm))
    
def axplot_ens_dpar_trajectories(ax, res, jm, jN):
    for j in range(res.pars['Npar']):
        ax.plot(res.dpar_gesges[jm,j,:,jN,:].T, color='C'+str(j))
    ax.hlines(res.pars['dpar'],0,res.dpar_gesges.shape[-1],color='gray',linestyle='--')  
    ax.set_title(r'$\lambda$ learning curves')
    ax.set_xlabel('it')
    
def axplot_ens_dpar_dist(ax, res, jm, jN):    
    for j in range(res.pars['Npar']):
        axplot_distribution(ax, res.dpar_gesges[jm,j,:,jN,:], color='C'+str(j))    
    ax.hlines(res.pars['dpar'],0,res.dpar_gesges.shape[-1],color='gray',linestyle='--') 
    ax.set_title(r'$\lambda$ learning curve distribution')
    ax.set_xlabel('it')
    
def axplot_ens_dpar_hist(ax, res, jm, jN):
    it = len(res.dpar_ges[jm][0])
    ax.hist(res.dpar_gesges[jm,0,:,jN,it-1], 20)
    ax.set_title(r'$\lambda$ distribution')
    ax.set_xlabel(r'$\lambda$')
    ax.vlines(res.dpar_gesges[jm,0,:,jN,it-1].mean(), 0, 8, colors='g')
    ax.vlines(res.pars['dpar'][0], 0, 8, colors='gray')
    
def axplot_ens_logL_vs_dpar(ax, res, jm, jN):
    ax2 = ax.twinx()
    it = np.array(res.dpar_ges[jm]).shape[1]
    axplot_logL_vs_nu(ax, res.pars, res.dpar_gesges[jm,0,:,jN,it-1], res.logL_gesges[jm,:,jN,0])
    axplot_logL_vs_nu(ax2, res.pars, res.dpar_gesges[jm,0,:,jN,it-1], res.logL_gesges[jm,:,jN,1], color='r')
    ax.set_xlabel(r'$\lambda$')
    ax2.tick_params(axis='y', colors='red')
    ax2.set_ylabel('')
    return ax2