# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = '1'

import sys
import time
import random
import collections
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from pathlib import Path

from punc.noise import *
from punc.plots import *
from punc.utils import * 
from punc.utils_train import *
from punc.datasets import get_ds
from punc.results import *
from punc.model import Model

from torch.utils.tensorboard import SummaryWriter

#%%    
if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not 'input_path' in locals(): input_path = set_input_path('../results/', 'test') #choose standard folder as input folder, if not otherwise specified
plot_path = input_path + 'plots/'
check_dirs(input_path, plot_path, replace=False)
exec(open(input_path+'input.py').read()) #run input file
pars['device'] = device


############################################
############################################
############################################

    #initialize functions, arrays, ... (outside of loop)


###
ds = get_ds(pars)
model = Model(ds, pars)
res = Results(pars, input_path)

#%%
for jN in range(pars['Nrun']):  #loop over different runs
    print('run #'+str(jN))
    plot_path_jN = plot_path + str(jN)
    
#%%    
############################################
############################################
############################################


    #initialize noise, networks, ...
    ###    
    model.initialize_data(pars)
    model.get_ytrain_etc(pars)
    if jN == 0:
        res.init_plot_arrays()
    
    model.adjust_data('dims')
    save_or_load_data(model, jN)
    model.adjust_data('device')
    res.init_run_results()
    
    
    # trainloaders
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(model.data['t_train'], model.data['y_train']), batch_size=pars['bs_train'], num_workers=0,  shuffle=True)
    trainiter = iter(trainloader)
    

#%%
############################################
############################################
############################################

    ## train PINN
    
    for jm in pars['model_vec']: #loop over different models
        model.itges, epoch_data = 0, 0 #total number of iterations, number of epochs
        writer, fpath = initialize_model_parameters(res, pars, comment, jN, jm)
        print_parameters(pars, input_path + 'pars.txt')
        ppath = writer.log_dir + '/'
        tm = time.time()
        
            
        ################################
        ################################
        ################################  start training
        model.init_nets(pars, jm, jN, device)
        model.set_nets_train()
        tlists = [ [] for j in range(7)]
        

#%%
        for j in range(model.itges, pars['Npinn']): #loop over iterations
            tpinn = time.time()
            tit = time.time()
            
            model.update_collocation_points(pars, ds, model.itges, device)          
            t_coll_batch, t_train_batch, y_train_batch, epoch_data, new_epoch = get_batches(model.t_colls, pars['dim'], trainiter, trainloader, epoch_data)            
                        
            #  calculate losses
            model.update_pde_weighting()
            res.losses_ges, res.losses_pde, res.losses_data, res.losses_prior, res.losses_repulsion, predictions_net_train = model.calculate_ensemble_losses(ds, t_coll_batch, t_train_batch, y_train_batch, pars)
               
            ### update network parameters
            for j_ens in range(pars['N_ensemble']):
                model.optimizers_pinn[j_ens].zero_grad()
                res.losses_ges[j_ens].backward()
                model.optimizers_pinn[j_ens].step()            
            
            model.itges += 1                
            ###  pars and plotting
            t5 = time.time()
            
            monitor_training(writer, pars, model.itges, jm, jN, ds, model, res, epoch_data, tpinn, ppath, device)
            
            if model.itges == pars['Npinn']:
                break
            if model.itges == pars['i_sched']:
                model.scheduler_pinn.step()


#%%    
        ####
        #### save pars    
        model.set_nets_eval()
        model.set_nets_device(device)
        res.t_model_ges[jm] = time.time() - tm
        
        
        
        
#%%    
        ####
        #### plotting
        print('')
        print('tm'+str(jm)+':', str(round(res.t_model_ges[jm],3)))
        model.evaluate_ensemble(jm, res)
        res.store_run_results(jm, jN)
        plot_ensemble(res, jm, jN, ppath)
        
        
        
#%%
    plot_ensemble_comparison(res, jN, ppath=ppath)

    #%%
    from punc.plots import *
    res.calculate_res_stats(jN)
    print_ensemble_comparison_table(res, jN, fpath=ppath, opt='flipped')
    print_ensemble_comparison_table(res, jN, fpath=input_path, opt='flipped')
    
    
    #### save res
    res.jN = jN
    fname = 'x'+str(pars['x_opt'])+'N'+ str(pars['N_ensemble']) +'r'+str(pars['Nrun'])
    filename = input_path + fname + '.dat'
    with open(filename,'wb') as f:
        pickle.dump(res, f)
    