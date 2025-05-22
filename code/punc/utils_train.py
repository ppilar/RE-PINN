# -*- coding: utf-8 -*-

import os
import shutil
import sys
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from pathlib import Path

from .utils import get_label
from .noise import get_noise
from .utils_data import *
from .plots import *



def save_or_load_data(model, jN):
    dpath = '../temp_data/' + model.pars['temp_folder'] + '/'
    dname = 'data_run'+str(jN)+'.pt'
    if model.pars['use_predefined_data'] == True and os.path.isfile(dpath + dname):
        with open(dpath + dname, 'rb') as f:
            model.data = pickle.load(f)
            update_par(model.pars, 'isample', model.data['isample'])
    else:
        Path(dpath).mkdir(parents=True, exist_ok=True)
        with open(dpath + dname, 'wb') as f:
            pickle.dump(model.data, f)           
            
        if jN == 0:
            print_parameters(model.pars, dpath+'pars.txt')




################
###############

def get_train_batches(trainiter, trainloader, epoch_data):
    #load next training batch
    
    new_epoch = -1
    try:
        train_batch = next(trainiter)
    except StopIteration:
        trainiter = iter(trainloader)
        train_batch = next(trainiter)
        epoch_data += 1
        new_epoch = 1
    return train_batch[0], train_batch[1], epoch_data, new_epoch


def get_batches(batch, dim, trainiter, trainloader, epoch_data):
    t_batch = batch
    t_train_batch, y_train_batch, epoch_data, new_epoch = get_train_batches(trainiter, trainloader, epoch_data)
    return t_batch, t_train_batch, y_train_batch, epoch_data, new_epoch
    

def sample_collocation_points(pars, N_coll, it = -1):
    if pars['collocation_base'] == 'grid':
        sN_coll = int((N_coll)**(1/pars['dim']))        
        bufs = [torch.linspace(pars['tmin_coll'][j], pars['tmax_coll'][j], sN_coll) for j in range(pars['dim'])]
        bufs2 = torch.meshgrid(*bufs)
        bufs3 = [bufs2[j].reshape(-1,1) for j in range(len(bufs2))]
        t_coll = torch.concatenate(bufs3,1)
    if pars['collocation_base'] == 'random':
        t_coll = torch.zeros(N_coll, pars['dim'])
        for j in range(pars['dim']):
            t_coll[:,j] = torch.rand(N_coll)*(pars['tmax_coll'][j] - pars['tmin_coll'][j]) + pars['tmin_coll'][j]
    if pars['collocation_base'] == 'Sobol':
        sampler = qmc.Sobol(d=pars['dim'], scramble=True)
        m = int(np.log2(N_coll))
        sample = sampler.random_base2(m=m)
        l_bounds = pars['tmin_coll']
        u_bounds = pars['tmax_coll']
        t_coll = torch.tensor(qmc.scale(sample, l_bounds, u_bounds)).float()
        
    return t_coll
  


def get_test_error(jm, ds, model, device, j_ens):  
    model.nets_pinn[j_ens].eval()    
    
    res_data_test, res_pde_test, y_net_test, x_net_test = model.calculate_residuals(model.data['t_test'], model.data['y_test'], ds, j_ens)
    res_data_train, res_pde_train, y_net_train, x_net_train = model.calculate_residuals(model.data['t_train'], model.data['y_train'], ds, j_ens)   
    
    model.nets_pinn[j_ens].train()
    model.data['t_train'].requires_grad = False    
    
    return res_data_train.detach().cpu(), res_data_test.detach().cpu(), res_pde_train.detach().cpu(), res_pde_test.detach().cpu(), y_net_train.detach().cpu(), y_net_test.detach().cpu(), x_net_test.detach().cpu()


def calculate_stat(stat):
    s_est = stat[:,:,:,-1]
    mu = s_est.mean(2)
    std = s_est.std(2)
    return (s_est, mu, std)


def calculate_stats(dpargesges, logLG_gesges, logLebm_gesges, rmse_gesges, fleval_gesges, dpar_true):
    #calculate statistics over multiple runs  

    dpar_true = np.array(dpar_true)
    
    dpar_buf = np.zeros(dpargesges.shape)
    for j in range(dpar_true.size):
        dpar_buf[:,j,:,:] = np.abs(dpargesges[:,j,:,:] - dpar_true[j])
    tdpar = calculate_stat(dpar_buf)
    
    tlogLG = calculate_stat(logLG_gesges)
    tlogLebm = calculate_stat(logLebm_gesges)
    tlogL = tlogLG
    for j in range(3):
        tlogL[j][1] = tlogLebm[j][1]
        tlogL[j][3] = tlogLebm[j][3]
    
    trmse = calculate_stat(rmse_gesges)
    tfl = calculate_stat(fleval_gesges)

    return tdpar, tlogL, trmse, tfl


def print_stats(itges, jm, res, ld_fac, lf_fac, epoch_data, nets_pinn, j_ens):
    #print current values to console
    
    if len(res.loss_data_ges[jm]) > 50:
        ldm = np.mean([l[j_ens] for l in res.loss_data_ges[jm][-50:]])
        lfm = np.mean([l[j_ens] for l in res.loss_pde_ges[jm][-50:]])
        ldstd = ld_fac*np.std([l[j_ens] for l in res.loss_data_ges[jm][-50:]])
        lfstd = lf_fac*np.std([l[j_ens] for l in res.loss_pde_ges[jm][-50:]])
        loss_str = ''
    else:
        ldm = res.loss_data_ges[jm][-1][j_ens]
        lfm = res.loss_pde_ges[jm][-1][j_ens]
        loss_str = ''
    
    if j_ens == 0:
        print("\r\r\r\r\r\r", end="")
    print(' itges' + str(itges)
          + ' dpar: ' + str(round(nets_pinn[j_ens].dpar[0].item(),4))
          + ' dpar2: ' + str(round(nets_pinn[j_ens].dpar[1].item(),4))
          + ' ld_fac: ' + str(round(ld_fac,4))
          + ' lf_fac: ' + str(round(lf_fac,4))
          + ' ld: ' + str(round(ldm,8))
          + ' lf: ' + str(round(lfm,8))
          + loss_str
          , end="")
    print('')




def monitor_training(writer, pars, itges, jm, jN, ds, model, res, epoch_data, tpinn, ppath, device):
    for jp in range(pars['Npar']):
        res.dpar_ges[jm][jp].append([net_pinn.dpar[jp].detach().item() for net_pinn in model.nets_pinn])
    res.loss_data_ges[jm].append([l.item() for l in res.losses_data])#.item())
    res.loss_pde_ges[jm].append([l.item() for l in res.losses_pde])
    res.loss_ges_ges[jm].append([l.item() for l in res.losses_ges])
    res.loss_rep_ges[jm].append([l.item() for l in res.losses_repulsion])

    rmse_data_ges_buf = []
    rmse_data_ges_buf2 = []
    rmse_pde_ges_buf = []
    rmse_pde_ges_buf2 = []
        
    if itges%pars['itest'] == 0:
        for j_ens in range(pars['N_ensemble']):
            res_data_train, res_data_test, res_pde_train, res_pde_test, y_net_train, y_net_test, x_net_test = get_test_error(jm, ds, model, device, j_ens)
            rmse_data_ges_buf.append(torch.sqrt(torch.mean((y_net_train.squeeze() - model.data['f_train'].cpu())**2)).detach().item())
            rmse_data_ges_buf2.append(torch.sqrt(torch.mean((y_net_test.squeeze() - model.data['f_test'].cpu())**2)).detach().item())
            rmse_pde_ges_buf.append(res_pde_train.detach().mean().sqrt().item())
            rmse_pde_ges_buf2.append(res_pde_test.detach().mean().sqrt().item())  
        
        res.rmse_data_ges[jm][0].append(rmse_data_ges_buf)
        res.rmse_data_ges[jm][1].append(rmse_data_ges_buf2)
        res.rmse_pde_ges[jm][0].append(rmse_pde_ges_buf)
        res.rmse_pde_ges[jm][1].append(rmse_pde_ges_buf2)  
              
    
    dpar_str = ['%.3f'%(net.dpar[0].detach().cpu().item()) for net in model.nets_pinn[:10]]
    lges_str = ['%.3f'%(l) for l in res.loss_ges_ges[jm][-1][:10]]
    print('\r' +'it'+str(itges) + ' lges ' + ' '.join(lges_str) +  ' dpars ' + ' '.join(dpar_str) + ' lf ' + str(pars['lf_fac_l']), end="")
    
    
    if model.itges%pars['iplot'] == 0 or itges == pars['Npinn']:
        with torch.no_grad():
            model.evaluate_ensemble(jm, res)
            if jm == pars['model_vec'][0] and itges == min(pars['iplot'], pars['Npinn']):
                res.store_run_data(model) 
            res.store_run_results(jm, jN)
            fig = plot_ensemble(res, jm, jN, ppath)
            writer.add_figure("ensemble", fig, itges)
            if pars['Npar'] > 1:
                fig = pairwise_plot(res.pars, res.dpar_gesges[:,:,:,jN,itges-1].transpose(0,2,1))
                writer.add_figure("dpars", fig, itges)
            
            
    #%% add new values to writer
    if itges % 20 == 0 or itges == 1 or itges == pars['Npinn']:
        writer.add_scalar("loss: ges", np.mean(res.loss_ges_ges[jm],1)[-1], itges)
        writer.add_scalar("loss: data", np.mean(res.loss_data_ges[jm],1)[-1], itges)
        writer.add_scalar("loss: data (std)", np.std(res.loss_data_ges[jm],1)[-1], itges)
        writer.add_scalar("loss: data (abs)", np.mean(np.abs(res.loss_data_ges[jm]),1)[-1], itges)
        writer.add_scalar("loss: pde", np.mean(res.loss_pde_ges[jm],1)[-1], itges)
        writer.add_scalar("loss: rep", np.mean(np.abs(res.loss_rep_ges[jm]),1)[-1], itges)
        writer.add_scalar("h1", model.h1, itges)
        writer.add_scalar("h2", model.h2, itges)
        if pars['learn_rhs'] == False:
            for j in range(pars['Npar']):
                buf = 'lambda ' + str(j)
                lambdas = [net.dpar[j].detach().item() for net in model.nets_pinn]
                writer.add_scalar(buf, np.abs(np.mean(lambdas) - pars['dpar'][j]), itges)
    
                    
            
            
        
        
#################
################# neural network definition

class Net(nn.Module): #map from t to x
    def __init__(self, pars, Uvec = [], Npar=4, fdrop = 0, input_dim = 1, output_dim = 1, bounds = (0,0,0,0), device = 'cpu'):
        super(Net, self).__init__()
        if Uvec == []:
            U = 40
            self.Uvec = [U]*5
        else:
            self.Uvec = Uvec        
            
        if pars['learn_pde_pars'] == 1:
            self.dpar = nn.Parameter(torch.normal(0.*torch.ones(Npar),1.*torch.ones(Npar)).abs())
        else:
            self.dpar = torch.tensor([0.01, 0, 0, 0]).float().to(device)        
           
        self.lb = torch.tensor(bounds[0]).float().to(device)
        self.ub = torch.tensor(bounds[1]).float().to(device)
        
        self.ylb = torch.tensor(bounds[2]).float().to(device)
        self.yub = torch.tensor(bounds[3]).float().to(device)
        
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for j, hdim in enumerate(self.Uvec):
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))
        
        self.dr = nn.Dropout(fdrop)
        if pars['activation'] == 'tanh':
            self.act = torch.tanh            
        elif pars['activation'] == 'relu':
            self.act = torch.relu
     
    def forward(self, X):
        #normalize
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input   
        
        for j, layer in enumerate(self.layers[:-1]):
            X = self.act(layer(X))
            
        X = self.dr(X)  
        X = self.layers[-1](X)
        
        X = 0.5*((X + 1.0)*(self.yub - self.ylb)) + self.ylb #reverse normalization of output

        return X












