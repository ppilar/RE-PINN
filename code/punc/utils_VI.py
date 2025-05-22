# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_KL(m1, m2, s1, s2):
    return torch.log(s1/s2) + (s2 + (m1 -m2)**2)/(2*s1**2) - 0.5


class StochasticLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(StochasticLinear, self).__init__()
        
        #mean and std of prior distribution of the weights
        self.pmean = torch.tensor(0.)
        self.pstd = torch.tensor(.1)
        
        #initialize parameters that define posterior distribution
        self.means = nn.Linear(dim_in, dim_out)
        self.stds = nn.Linear(dim_in, dim_out)
        
        
    def get_std(self, std0):
        return torch.log(1 + torch.exp(std0))
        
        
    def sample_weights(self, device):
        self.weights = self.means.weight + self.get_std(self.stds.weight) * (torch.normal(torch.zeros(self.means.weight.shape), torch.ones(self.stds.weight.shape))).to(device)
        self.biases = self.means.bias + self.get_std(self.stds.bias) * (torch.normal(torch.zeros(self.means.bias.shape), torch.ones(self.stds.bias.shape))).to(device)
        
        
    def get_KL_term(self):
        KL = get_KL(self.pmean, self.means.weight, self.pstd, self.get_std(self.stds.weight)).sum()
        KL = KL + get_KL(self.pmean,self.means.bias, self.pstd, self.get_std(self.stds.bias)).sum()
        return KL
        
    
    def forward(self, X, sample_weights=True):
        if sample_weights:
            self.sample_weights(device = X.device)
        
        if X.ndim == 1:
            X = X.unsqueeze(1)
        
        X = X @ self.weights.T + self.biases.repeat(X.shape[0],1)        
        
        return X
    

class StochasticNet(nn.Module): #map from t to x
    def __init__(self, Uvec = [20]*2, Npar=3, fdrop = 0, input_dim = 1, output_dim=1, bounds = (0,0,0,0), device = 'cpu'):
        super(StochasticNet, self).__init__()
        self.device = device
        self.itges = 0
        
        self.Uvec = Uvec
        self.init_layers(input_dim, output_dim)        
        self.act = torch.tanh
        
        self.dpar = StochasticLinear(1, 4)
        
        self.lb = torch.tensor(bounds[0]).float().to(device)
        self.ub = torch.tensor(bounds[1]).float().to(device)
        
        self.ylb = torch.tensor(bounds[2]).float().to(device)
        self.yub = torch.tensor(bounds[3]).float().to(device)
        
    
    def sample_dpar(self):
        self.dpar.sample_weights(device=self.device)
        return self.dpar.weights
    
    
    def init_layers(self, input_dim, output_dim):
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for j, hdim in enumerate(self.Uvec):
            self.layers.append(StochasticLinear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(StochasticLinear(current_dim, output_dim))

        
    def get_KL_term(self):
        KL = 0
        for l in self.layers:
            KL = KL + l.get_KL_term()  
        KL = KL + self.dpar.get_KL_term()
        return KL
    
    def generate_samples(self, X, Ns=1):
        y_samples = torch.zeros(Ns, X.shape[-1])
        dpar_samples = torch.zeros(Ns, 4)
        for j in range(Ns):
            y_samples[j] = self.forward(X).squeeze().detach()
            dpar_samples[j] = self.sample_dpar().squeeze().detach()
        return y_samples, dpar_samples
     
    def forward(self, X):        
        
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input 
        
        for j, layer in enumerate(self.layers[:-1]):
            X = self.act(layer(X))            
        X = self.layers[-1](X)
        
        X = 0.5*((X + 1.0)*(self.yub - self.ylb)) + self.ylb #reverse normalization of output
        return X


class ResultsVI():
    def __init__(self, pars):
        self.xnp_gesges = np.zeros((1, pars['num_samples'], pars['Nrun'], *pars['f_plot_shape']))
        self.dpar_gesges = np.zeros((1, pars['Npar'], pars['num_samples'], pars['Nrun'], 1))
        self.data = [[] for j in range(pars['Nrun'])]
        self.pars = pars.copy()
        self.pars['N_ensemble'] = pars['num_samples']
        
    def store_run_results(self, data, preds, samples, jN):
        self.xnp_gesges[0,:,jN,:,:] = preds.cpu().numpy()
        self.dpar_gesges[0, :, :, jN, 0] = samples.T
        self.data[jN] = data
        
        
        



def get_likelihood_term(SNet, pars, x_train, y_train, x_coll=None, device='cpu'):
    if x_coll is not None:
        x_ges = torch.cat((x_train, x_coll))
    else:
        x_ges = x_train
    
    logL_term = 0
    f_nets = []
    dpar_nets = []
    for k in range(pars['NMC']):
        x_buf = x_ges if pars['use_x_coll'] else x_train
        f_nets.append(SNet(x_buf).squeeze())
        logL_term = logL_term + torch.sum(((f_nets[k][:pars['N_train']] - y_train)**2))/pars['NMC']
        dpar_nets.append(SNet.sample_dpar())
    f_nets = torch.vstack(f_nets)
    return logL_term, f_nets[:,pars['N_train']:], x_ges[pars['N_train']:], dpar_nets


    
def axplot_distribution(ax, fmean, fstd, t_plot=-1, color='#1f77b4'):
    if type(t_plot) == int:
        t_plot = np.arange(fstd.shape[-1])
    ax.plot(t_plot, fmean, label='mean', zorder=9)
    ax.fill_between(t_plot, fmean + fstd, fmean - fstd, alpha=0.5, color=color)
    ax.fill_between(t_plot, fmean + 2*fstd, fmean - 2*fstd, alpha=0.4, color=color)
    ax.fill_between(t_plot, fmean + 3*fstd, fmean - 3*fstd, alpha=0.3, color=color)

    
def plot_SNet(SNet, x_train, y_train, ppath, savefig = True):
    device = x_train.device
    
    Nf = 100
    Nx = 200
    x_plot = torch.linspace(x_train.min(), x_train.max()+5,Nx).to(device)
    y_plots = torch.zeros(Nf, Nx)
    for j in range(Nf):
        y_plots[j] = SNet(x_plot).squeeze().detach()
    

    if device != 'cpu':
        x_train = x_train.to('cpu')
        y_train = y_train.to('cpu')
        x_plot = x_plot.to('cpu')
        y_plots = y_plots.to('cpu')
    
    
    fig, axs = plt.subplots(1,2, figsize=(24,8))
    axs[1].plot(x_plot, y_plots.T[:, :10], linewidth=1)
    axs[1].scatter(x_train, y_train, color='k', zorder=9)
    axs[1].set_xlabel('t')    
    axs[1].set_ylabel('x(t)')
    
    axplot_distribution(axs[0], y_plots.mean(0), y_plots.std(0), t_plot = x_plot)
    axs[0].scatter(x_train, y_train, color='k', zorder=9)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x(t)')
    
    fig.suptitle('it=%.i'%(SNet.itges))
    
    if savefig:
        fig.savefig(ppath + 'BNN_variational.pdf', bbox_inches='tight')
        
    plt.show()
    return fig



