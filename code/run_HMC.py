# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS, Predictive, RandomWalkKernel

from punc.utils_HMC import *
from punc.plots import pairwise_plot

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'

#%% load data and visualize

# Set random seed for reproducibility
s = 0
np.random.seed(s)
pyro.set_rng_seed(s)

pars = {}
#%%
fpath = '../results/test/'
fname = 'x5N10r1.dat'


with open(fpath + fname, 'rb') as f:
    res = pickle.load(f)
    #res.pars['MC_kernel'] = 'standard'
    res.pars['MC_kernel'] = 'nuts'    
    
res.pars['use_pde_loss'] = True#False


#%% define model
print(fpath+fname)

ppath = fpath + 'plots/'

sol_ges = []
rhs_ges = []

if not 'real_data' in res.pars:
    res.pars['real_data'] = 0
res.pars['num_samples'] = 5000
res_mc = ResultsMC(res.pars)
pars['num_chains'] = 1
#%% run HMC
for jN in range(res.pars['Nrun']):
    #with torch.autograd.set_detect_anomaly(True):
    samples, preds = get_MC_samples(res.pars, res.data[jN], device)    
    res_mc.store_run_results(res.data[jN], preds, samples, jN)
    
    #%%
    pname = 'HMC_pde_'+str(res.pars['use_pde_loss'])+'_N_'+str(res.pars['num_samples']) + '_chains_' + str(res.pars['num_chains']) +'_jN' + str(jN) +'_testdata'
    plot_results(res_mc, jN, ppath, pname)
    pairwise_plot(res_mc.pars, res_mc.dpar_gesges[:,:,:,jN].transpose((0,2,1)), opt='only_mc', ppath = ppath + 'jN' + str(jN)+'_')
    
#%%
with open(fpath + 'mc_'+fname,'wb') as f:
    pickle.dump(res_mc, f)