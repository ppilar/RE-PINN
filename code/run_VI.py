import numpy as np
import sys
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from punc.utils import *
from punc.utils_VI import *

if not 'rand_init' in locals():
    rand_init, s = init_random_seeds(s=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device='cpu'


ds_opt = 1
if ds_opt == 1:
    fpath = '../results/res_ds1_N_ensemble/ds1_N_ensemble5/'
    fname = 'x1N5r5.dat' 
    bounds = (0,10,0,20)
    jN = 0
elif ds_opt == 5:
    fpath = '../results/res_ds5_N_train10v2_gap_fn1/'
    fname = 'x5N25r5.dat'
    bounds = (0,20,-1,1)
    jN = 3
with open(fpath + fname, 'rb') as f:
    res = pickle.load(f)

#%%

if ds_opt == 1:
    def loss_f(t, x, dpars):
        lf = 0
        #dpar=[0.3]
        for j in range(x.shape[0]):            
            dpar = dpars[j]
            xdot = torch.autograd.grad(x[j].sum(), t, create_graph=True)[0]        
            omega = dpar[0]
            lf = lf + ((xdot - omega * x[j]) ** 2).mean()/x.shape[0]
    
        return lf    
    
    x_coll = torch.linspace(0,15,100, device=device, requires_grad = True)
    
elif ds_opt == 5:
    def loss_f(t, x, dpars):
        lf = 0
        #dpar=[1, 0.1]
        for j in range(x.shape[0]):
            dpar = dpars[j]
            xdot = torch.autograd.grad(x[j].sum(), t, create_graph=True)[0]
            xdot2 = torch.autograd.grad(xdot.sum(), t, create_graph=True)[0]
        
            omega, zeta = dpar[:2]
            lf = lf + ((xdot2 + 2 * zeta * omega * xdot + omega**2 * x[j]) ** 2).mean()/x.shape[0]
    
        return lf
    
    x_coll = torch.linspace(0,25,125, device=device, requires_grad = True)


 
def get_lambda_pde(it):
    if it > 15000:
        return 5 
    return 1

def get_lambda_KL(it):
    if it > 18000:
        return 0.0001
    return 0

#%% settings
if not 'pars' in locals():
    pars = dict()
init_par(pars, 'ds', ds_opt)
init_par(pars, 'use_x_coll', True)
init_par(pars, 'Nit', 20000)
init_par(pars, 'NMC', 5)
init_par(pars, 'lr', 0.01)
init_par(pars, 'lambda_KL', 1)
ppath = '../plots/'


#%% initialize data and models
x_train, y_train = torch.tensor(res.data[jN]['t_train'], device=device).float().squeeze(), torch.tensor(res.data[jN]['y_train'], device=device).float().squeeze()
pars['N_train'] = N_train = x_train.shape[0]
SNet = StochasticNet(Uvec = res.pars['Uvec_pinn'], bounds=bounds, device=device).to(device)
opt = optim.Adam(SNet.parameters(), lr=pars['lr'])
if not 'comment' in locals(): comment = ""
#writer, ppath = initialize_writer("../runs/", comment=comment)
writer = initialize_writer("../runs/", comment=comment)
print_parameters(pars, writer.log_dir + '/pars.txt')



#%% load SNet to continue training, when desired
# fname = 'x5_it25039_lambda_final'
# filename = '../results/'+ fname + '.dat'
# with open(filename,'rb') as f:
#     SNet, res, pars = pickle.load(f)
# opt = optim.Adam(SNet.parameters(), lr=pars['lr'])



#%% plot SNet
plot_SNet(SNet, x_train, y_train, ppath+'it0_')

losses = []
t1_count = 0
#%% training

t = tqdm(range(pars['Nit']), desc='Training SNN:')
for j in t:
    #log likelihood term
    t0 = time.time()
    logL_term, y_nets, x_ges, dpars = get_likelihood_term(SNet, pars, x_train, y_train, x_coll=x_coll, device=device)
    t0 = time.time() - t0
        
    #else: #with weight-space prior
    KL_term = SNet.get_KL_term().mean()
    pde_term = loss_f(x_coll, y_nets, dpars)
    
    pars['lambda_pde'] = get_lambda_pde(SNet.itges)
    pars['lambda_KL'] = get_lambda_KL(SNet.itges)
    loss = logL_term + pars['lambda_pde']*pde_term + pars['lambda_KL']*KL_term
    
    t4 = time.time()
    opt.zero_grad()
    loss.backward()
    opt.step()
    t4 = time.time() - t4
    
    #################
    #################
    #gather statistics
    
    t.set_description('i=%.i, Loss=%.2f, l=%.2f, sl=%.2f'%(SNet.itges, loss, SNet.dpar.means.weight[0], SNet.dpar.get_std(SNet.dpar.stds.weight[0])))
    losses.append(loss.item())
    SNet.itges += 1
    
    if SNet.itges%20 == 0:
        writer.add_scalar("loss: ges", loss.item(), SNet.itges)
        writer.add_scalar("loss: likelihood", logL_term.item(), SNet.itges)
        writer.add_scalar("loss: KL prior", KL_term.mean().item(), SNet.itges)
    if SNet.itges%1000 == 0:
        fig = plot_SNet(SNet, x_train, y_train, ppath)
        writer.add_figure("SNet", fig, SNet.itges)

    
    
    
#%%
plot_SNet(SNet, x_train, y_train, ppath)

#%% save net
fname_save = 'x'+str(ds_opt)+'_it' + str(SNet.itges) + '_lambda'
filename = '../results/'+ fname_save + '.dat'
with open(filename,'wb') as f:
    pickle.dump([SNet, res, pars], f)


    
#%%
Nx = 200
Ns = 100
x_plot = torch.linspace(x_train.min(), x_train.max()+5,Nx)
y_nets, dpars = SNet.generate_samples(x_plot.to(device), Ns)
y_nets = y_nets.unsqueeze(-1)
dpars = dpars[:,:2]

pars_VI = res.pars.copy()
pars_VI['num_samples'] = Ns
res_VI = ResultsVI(pars_VI)
res_VI.store_run_results(res.data[jN], y_nets, dpars, jN)

with open(fpath + 'VI_'+fname,'wb') as f:
    pickle.dump(res_VI, f)