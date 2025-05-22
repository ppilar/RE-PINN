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
from torchdiffeq import odeint


from punc.utils import init_par

   
    
########################
########################

class ResultsMC():
    def __init__(self, pars):
        self.xnp_gesges = np.zeros((1, pars['num_samples'], pars['Nrun'], *pars['f_plot_shape']))
        self.dpar_gesges = np.zeros((1, pars['Npar'] + pars['output_dim'], pars['num_samples'], pars['Nrun']))
        self.data = [[] for j in range(pars['Nrun'])]
        self.pars = pars.copy()
        self.pars['N_ensemble'] = pars['num_samples']
        
    def store_run_results(self, data, preds, samples, jN):
        self.xnp_gesges[0,:,jN,:,:] = preds['obs'].cpu().numpy()
        buf = torch.stack([samples[key] for key in samples], 0)
        if buf.ndim == 3:
            buf = buf.squeeze()
        self.dpar_gesges[0, :, :, jN] = buf.cpu()
        self.data[jN] = data
        

###################
###################


def plot_data(data):    
    fig, axs = plt.subplots(1,2,figsize=(9,4))
    for j in range(data['y_train'].shape[1]):
        axs[0].scatter(data['t_train'].cpu(), data['y_train'][:,j].cpu())
        axs[0].scatter(data['t_test'].cpu(), data['y_test'][:,j].cpu(), c='r')

def plot_results(res_mc, jN, ppath='', pname='HMC'):
    from .plots import axplot_distribution, axplot_ens_fdist, axplot_ens_fx
    
    fig, axs = plt.subplots(1,2, figsize=(18,8))
    axplot_ens_fdist(axs[0], res_mc, 0, jN, dist_opt='mean')
    if res_mc.pars['N_ensemble'] < 500:
        axplot_ens_fx(axs[1], res_mc, 0, jN)    
    
    axs[0].set_ylim(res_mc.pars['ylim'])
    axs[1].set_ylim(res_mc.pars['ylim'])
    
    fig.savefig(ppath+pname+'.pdf', bbox_inches='tight')
    
    
#%%    
def get_ML_preds(t, model, posterior_samples):
    preds = {}
    preds['obs'] = model.generate_solution(t, posterior_samples)
    return preds

def get_MC_samples(pars, data, device='cpu'):
    for key in data:
        if type(data[key]) == np.ndarray:
            data[key] = torch.Tensor(data[key]).float()
            
    init_par(pars, 'num_chains', 1)
    init_par(pars, 'MC_kernel', 'standard')
    init_par(pars, 'num_samples', 200)
    init_par(pars, 'sigma', np.sqrt(pars['lr_fac']/2))
    
    if not 't_train_rhs' in data:
        data['t_train_rhs'] = torch.tensor([0])
        data['y_train_rhs'] = torch.tensor([0])
    
    
    
    # Hook function to track iterations
    iteration_counter = [0]  
    def hook_fn(kernel, samples, stage, i):
        iteration_counter[0] += 1
    
    
    model = ML_model(pars, device)
    if pars['MC_kernel'] == 'nuts':
        kernel = NUTS(model, jit_compile=False)
    else:
        kernel = RandomWalkKernel(model, init_step_size = 0.1, target_accept_prob = 0.2)
        
    
    if pars['x_opt'] == 7:
        #initial_pars = {'f0': torch.tensor([40.]), 'f1': torch.tensor([15.]), 'l0': torch.tensor([1.]), 'l1': torch.tensor([0.02]), 'l2': torch.tensor([2.]), 'l3': torch.tensor([0.025])}
        initial_pars = {'f0': torch.tensor([3.5]), 'f1': torch.tensor([0.5]), 'l0': torch.tensor([2.]), 'l1': torch.tensor([2.]), 'l2': torch.tensor([2.]), 'l3': torch.tensor([2.])}
        #initial_pars = {'f0': torch.tensor([10.]), 'f1': torch.tensor([5.]), 'l0': torch.tensor([1.5]), 'l1': torch.tensor([1.]), 'l2': torch.tensor([3.]), 'l3': torch.tensor([1.])}
        #initial_pars = {'f0': torch.tensor([0.]), 'l0': torch.tensor([1.]), 'l1': torch.tensor([0.2])}
        mcmc = MCMC(kernel, initial_params = initial_pars, warmup_steps = min(1000, pars['num_samples']), num_samples=pars['num_samples'], num_chains=pars['num_chains'], hook_fn=hook_fn)
    else:
        mcmc = MCMC(kernel, warmup_steps = min(1000, pars['num_samples']), num_samples=pars['num_samples'], num_chains=pars['num_chains'], hook_fn=hook_fn)


    t_train_ges = (data['t_train'].to(device).squeeze(), data['t_train_rhs'].to(device).squeeze())
    y_train_ges = (data['y_train'].to(device).squeeze(), data['y_train_rhs'].to(device).squeeze())
    mcmc.run(t_train_ges, y_train_ges)
    
    samples = mcmc.get_samples()
    if not 'f_plot' in data:
        data['f_plot'] = get_ML_preds(data['t_plot'].squeeze(), model=model, posterior_samples = {'f0': 10., 'f1': 5., 'l0': 1.5, 'l1': 1., 'l2': 3., 'l3': 1.})['obs'].T
    preds = get_ML_preds(data['t_plot'].to(device).squeeze(), model=model, posterior_samples = samples)

    if preds['obs'].ndim == 2:
        preds['obs'] = preds['obs'].unsqueeze(-1)
    return samples, preds

    

# Define the derivative function: f'(x) = a * f(x)
class ExponentialODE(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, t, f):
        return self.a * f
    
# Define the Lotka-Volterra ODE system
class LotkaVolterraODE(torch.nn.Module):
    def __init__(self, alpha, beta, gamma, delta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, t, z):
        x, y = z[0], z[1]
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return torch.stack([dxdt, dydt])

    

class ML_model(PyroModule):
    def __init__(self, pars, device):
        super().__init__()
        self.pars = pars
        self.device = device
    
    
    def get_lvals(self):
        lvals = {}
        if self.pars['x_opt'] == 1:
            means = torch.tensor([0., 0.], device=self.device)
            stds = torch.tensor([5., 1.], device=self.device)
            lvals['f0'] = pyro.sample("f0", dist.Uniform(-10,10)).unsqueeze(0)
            lvals['l0'] = pyro.sample("l0", dist.Uniform(-10,10)).unsqueeze(0)
        elif self.pars['x_opt'] == 5:
            lvals['f0'] = pyro.sample("f0", dist.Uniform(0,1.5)).unsqueeze(0)
            lvals['l0'] = pyro.sample("l0", dist.Uniform(0,3)).unsqueeze(0)
            lvals['l1'] = pyro.sample("l1", dist.Uniform(0,0.9)).unsqueeze(0)
        elif self.pars['x_opt'] == 7:
            #pars = torch.tensor([40, 15, 1., 0.02, 2., 0.025])
            #off = torch.tensor([10, 5, 0.5, 0.01, 0.5, 0.01])
            pars = torch.tensor([3.5, 1., 2., 2., 2., 2.])
            off = torch.tensor([1., 1., 2., 2., 2., 2.])
            
            for j, key in enumerate(['f0', 'f1', 'l0', 'l1', 'l2', 'l3']):
                lvals[key] = pyro.sample(key, dist.Uniform(pars[j]-off[j], pars[j]+off[j]))
                if lvals[key].ndim == 0:
                    lvals[key] = lvals[key].unsqueeze(0)
        return lvals
    
    def generate_solution(self, t, lvals):
        if self.pars['x_opt'] == 1:
            solution = (lvals['f0'].unsqueeze(1)*torch.exp(lvals['l0'].unsqueeze(1)*t.unsqueeze(0)))
            #ode_func = ExponentialODE(lvals['l0']).to(self.device)
            #solution = odeint(ode_func, lvals['f0'], t).T
            return solution
        if self.pars['x_opt'] == 5:
            f0, omega, zeta = [lvals[key].unsqueeze(1) for key in lvals]
            return f0*(torch.exp(-zeta * omega * t) * torch.sin(omega * torch.sqrt(1 - zeta**2) * t.unsqueeze(0)))
            
        if self.pars['x_opt'] == 7:
            lv_model = LotkaVolterraODE(lvals['l0'], lvals['l1'], lvals['l2'], lvals['l3'])
            solution = odeint(lv_model, torch.stack((lvals['f0'], lvals['f1']),0), t)
            if solution.ndim > 3:
                solution = solution.squeeze()
            return solution.permute(2,0,1)
    
    def forward(self, x, y=(None, None)):        
        x, _ = x
        y, _ = y
            
        lvals = self.get_lvals()
        sigma = torch.tensor(self.pars['sigma'], device=x.device).float()
        mu = self.generate_solution(x, lvals).squeeze()
            
        
        # Sampling model
        with pyro.plate("data", mu.view(-1).shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu.view(-1), sigma), obs=y.view(-1))
            
        return (mu, None)