# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from sklearn.neighbors import KernelDensity

from .utils import *
from .utils_train import *
from .noise import get_noise
from .utils_data import *
from .utils_repulsion import *

#class to store results
class Model():
    def __init__(self, ds, pars):
        self.initialize_functions(ds, pars)
        self.x_opt = pars['x_opt']
        self.N_ensemble = pars['N_ensemble']
        self.learn_rhs = pars['learn_rhs']
        self.pars = pars
        self.h1 = 0
        self.h2 = 0
        self.device = self.pars['device']
        
    def initialize_functions(self, ds, pars):
        self.fx = ds.get_fx()
        self.fn = get_noise(pars['n_opt'], pars['nfac']) #function to generate noise
        self.fmeq = ds.get_meas_eq() #function defining measurement equation
        self.flf = ds.get_loss_f() #pde loss
        self.fld0 = ds.get_loss_d() #data loss
        self.frhs = ds.get_frhs()
        
    def initialize_data(self, pars):
        self.data = {}
        self.data['isample'] = pars['isample']
    
        #initialize training and test data
        if pars['x_opt'] == 7 and pars['real_data'] == 1:
            self.data['t_train'], self.data['y_train'], self.data['t_test'], self.data['y_test'] = load_hare_lynx_data(pars, opt='torch')
                        
            f = 10.
            self.ymax = torch.tensor(80.)/f
            self.data['y_train'] /= f
            self.data['y_test'] /= f
            self.data['t_train'] -= 1900
            self.data['t_test'] -= 1900
            
            
            
            self.data['f_train'] = self.data['y_train']
            self.data['f_test'] = self.data['y_test']
            
            self.ymin = torch.tensor(0.)
            self.ymax = torch.tensor(80.)/f
            
            pars['tmin'] = (0,)#(1900.,)
            pars['tmax'] = (20,)#(1920.  ,)     
            pars['tmin_coll'] = (0,)#(1900.,)
            pars['tmax_coll'] = (20,)#((1920.,)
            pars['ylim'] = 0., 80/f
            
        elif pars['x_opt'] in [1,2,3,5,7]:
            self.data['t_train'], self.data['f_train'], self.data['t_test'], self.data['f_test'] = self.generate_solution(pars)
            if pars['learn_rhs'] == True:
                self.data['t_train_rhs'], self.data['f_train_rhs'], self.data['t_test_rhs'], self.data['f_test_rhs'] = self.generate_rhs_solution(pars)
        if pars['x_opt'] == 51:
            self.data['t_train'], self.data['f_train'], self.data['t_test'], self.data['f_test'] = self.generate_solution(pars)
            if pars['load_data'] == True:
                t_train, f_train = np.load('../data/1dreg.npy')
                self.data['t_train'] = torch.tensor(t_train).float()
                self.data['f_train'] = torch.tensor(f_train).float()
        if pars['x_opt'] == 52:
            buf = np.load('../data/wilson_1dreg.npz')
            self.data['t_train'] = torch.tensor(buf['x']).float()
            self.data['f_train'] = 0.1*torch.tensor(buf['y']).float().squeeze()
            self.data['t_test'] = torch.tensor(buf['x_']).float()
            self.data['f_test'] = 0.1*torch.tensor(buf['y_']).float().squeeze()
        elif pars['x_opt'] == 102:
            self.data['t_train'], self.data['f_train'], self.data['t_test'], self.data['f_test'] = load_advection_data(pars)
        elif pars['x_opt'] == 103:
            self.data['t_train'], self.data['f_train'], self.data['t_test'], self.data['f_test'] = load_advection_data(pars)
        
        pars['N_train'] = self.data['t_train'].shape[0]
        pars['N_test'] = self.data['t_test'].shape[0]        
        if pars['learn_rhs'] == True:
            pars['N_train_rhs'] = self.data['t_train_rhs'].shape[0]
        
        self.t_colls = [-1]*pars['N_ensemble']
        self.dim = pars['dim']
        
        
        #arrays for plotting
        self.data['t_plot'], self.pars['t_plot_shape'] = get_t_plot(self.pars['tmin_coll'], self.pars['tmax_coll'], self.pars['dim'])
        if type(self.fx) is not int:
            self.data['f_plot'] = self.fx(self.data['t_plot'])
            if pars['learn_rhs']:
                self.data['f_plot_rhs'] = self.frhs(self.data['t_plot'])
        else:
            self.data['t_plot'] = self.data['t_test']
            self.data['f_plot'] = self.data['f_test']        
            self.pars['t_plot_shape'] = pars['tshape']
        
        self.pars['f_plot_shape'] = (*self.pars['t_plot_shape'], self.pars['output_dim'])

        
        
        print('test')
        
        
    def generate_solution(self, pars):
        #if pars['t_opt'] == 'linspace':
        if pars['t_opt'] == 'gap':
            if pars['x_opt'] == 5:
                t_train = torch.cat((torch.linspace(pars['tmin'][0],6,5), torch.linspace(15,pars['tmax'][0], pars['N_train']-5)))                
            elif pars['x_opt'] == 7:
                t_train = torch.cat((torch.linspace(0,4,7), torch.linspace(8,10, pars['N_train']-7)))
        else:
            t_train = torch.linspace(pars['tmin'][0], pars['tmax'][0], pars['N_train'])
        
        #t_test = torch.linspace(pars['tmax'][0], pars['tmax_coll'][0], pars['N_test'])
        t_test = torch.linspace(pars['tmin'][0], pars['tmax'][0], pars['N_test'])
        
        x_train = self.fx(t_train)
        f_train = self.fmeq(x_train)
        x_test = self.fx(t_test)
        f_test = self.fmeq(x_test)
        
        
        return t_train, f_train, t_test, f_test
    
    def generate_rhs_solution(self, pars):
        t_train_rhs = torch.linspace(pars['tmin_rhs'][0], pars['tmax_rhs'][0], pars['N_train_rhs'])        
        f_train_rhs = self.frhs(t_train_rhs)
        t_test_rhs = torch.linspace(pars['tmin_rhs'][0], pars['tmax_rhs'][0], pars['N_test_rhs'])
        f_test_rhs = self.frhs(t_test_rhs)
        return t_train_rhs, f_train_rhs, t_test_rhs, f_test_rhs
        
    #add noise to function values
    def add_noise(self, f):
        N = f.shape[0]
        d = f.shape[1] if f.ndim > 1 else 1
        noise = self.fn.sample(N*d).reshape((N,d)).squeeze()
        y = f + noise
        return y, noise
    
    #adds noise to train and test data
    def get_ytrain_etc(self, pars): 
        init_par(pars, 'lr_fac', 2*self.fn.sig**2)
                
        if pars['real_data'] != 1:
            self.data['y_train'], self.data['noise_train'] = self.add_noise(self.data['f_train'])
            self.data['y_test'], self.data['noise_test'] = self.add_noise(self.data['f_test'])
            self.ymin, self.ymax = torch.min(self.data['y_train']), torch.max(self.data['y_train'])
            Nd = self.data['y_train'].numel()            
            
            if pars['learn_rhs'] == True:
                self.data['y_train_rhs'], self.data['noise_train_rhs'] = self.add_noise(self.data['f_train_rhs'])
                self.data['y_test_rhs'], self.data['noise_test_rhs'] = self.add_noise(self.data['f_test_rhs'])
                self.ymin_rhs, self.ymax_rhs = torch.min(self.data['y_train_rhs']), torch.max(self.data['y_train_rhs'])
                
                Nd += self.data['y_train_rhs'].numel()
        else:
            Nd = self.data['y_train'].numel()
            
        
        
    def adjust_data(self, opt='dims'):
        for key in self.data:
            if type(self.data[key]) == torch.Tensor:
                if opt == 'dims':
                    if self.data[key].ndim == 1:
                        self.data[key] = self.data[key].unsqueeze(1)
                if opt == 'device':
                    self.data[key] = self.data[key].float().to(self.device)
            
        
    def set_nets_train(self):
        for net in self.nets_pinn + self.nets_rhs:
            net.train()
            
    def set_nets_eval(self):
        for net in self.nets_pinn + self.nets_rhs:
            net.eval()        
        
    def set_nets_device(self, device):
        for net in self.nets_pinn + self.nets_rhs:
            net.to(device)
            
            
    # to use the same initialization for the different models and isolate the effect of the repulsion
    def save_or_load_initialization(self, pars, j, jm, opt='pinn'):
        buf_net = self.nets_pinn[-1] if opt == 'pinn' else self.nets_rhs[-1]
        buf_name = 'pinn' if opt == 'pinn' else 'rhs'
        
        if jm == pars['model_vec'][0]:
            torch.save(buf_net.state_dict(), "../temp_data/net_"+buf_name+str(j)+".pt")
        else:
            buf_net.load_state_dict(torch.load("../temp_data/net_"+buf_name+str(j)+".pt"))
            
    def init_net(self, nets, pars, Uvec, bounds, j, jm, opt, device):        
        nets.append(Net(pars, Uvec, fdrop=pars['fdrop_pinn'], input_dim = pars['dim'], output_dim = pars['output_dim'], bounds = bounds, device = device).to(device))  
        if not pars['learn_pde_pars']:
            nets[-1].dpar.requires_grad = False
            nets[-1].dpar[:pars['Npar']] = torch.tensor(pars['dpar']).float().to(device)
        self.save_or_load_initialization(pars, j, jm, opt=opt)
        return list(nets[-1].parameters())
    
    def init_nets(self, pars, jm, jN, device):
        bounds_pinn = (pars['tmin_coll'], pars['tmax_coll'], self.ymin.item(), self.ymax.item())    
        if pars['learn_rhs'] == 1: bounds_rhs = (pars['tmin_coll'], pars['tmax_coll'], self.ymin_rhs.item(), self.ymax_rhs.item())
        self.nets_pinn, self.nets_rhs, self.optimizers_pinn, self.schedulers_pinn = [[] for j in range(4)]
        
        for j in range(pars['N_ensemble']):
            optim_pars = self.init_net(self.nets_pinn,  pars, pars['Uvec_pinn'], bounds_pinn, j, jm, 'pinn', device)
            if pars['learn_rhs'] == True:
                optim_pars += self.init_net(self.nets_rhs, pars, pars['Uvec_rhs'], bounds_rhs, j, jm, 'rhs', device)
            self.optimizers_pinn.append(optim.Adam(optim_pars, lr=pars['lr_pinn'], weight_decay=pars['weight_decay']))
            self.schedulers_pinn.append(torch.optim.lr_scheduler.ExponentialLR(self.optimizers_pinn[-1], gamma=0.3))
        
    def update_collocation_points(self, pars, ds, itges, device='cpu'):
        for j in range(pars['N_ensemble']):
            self.t_colls[j] = sample_collocation_points(pars, pars['N_coll'], self.itges).to(device)
    
    def calculate_prediction(self, t, ds, j_ens):
        if t.is_leaf:
            t.requires_grad = True
        x_net = self.nets_pinn[j_ens].forward(t)
        ds.set_temp(t, x_net, self.nets_pinn[j_ens].dpar)
        y_net = self.fmeq(x_net)
        return y_net, x_net
    
    def calculate_residuals(self, t, y, ds, j_ens):
        y_net, x_net = self.calculate_prediction(t, ds, j_ens)
        data_residuals =  self.get_data_residuals(y, y_net).float()
        pde_residuals = self.get_pde_residuals(t, x_net, j_ens)
        return data_residuals, pde_residuals, y_net, x_net
    
    def get_logL(self, f):
        fvals = f(self.t_train.cpu()).to(self.y_train.device)
        return torch.sum(-(fvals - self.y_train)**2/(2*self.pars['nfac']**2) + 0.5*torch.log(torch.tensor(2*torch.pi*self.pars['nfac']**2)))
    
    
    
    def get_data_residuals(self, y, y_net):        
        res = (y.squeeze() - y_net.squeeze())
        return res.flatten()
    
    def get_pde_residuals(self, t, x, j_ens):
        if self.learn_rhs == True:
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            fnet = self.nets_rhs[j_ens](t)
            return (xdot - fnet)**2
        else:
            return self.flf(t, x, self.nets_pinn[j_ens].dpar)         
    
        
    def get_losses_repulsion(self, pars, predictions_net_train, predictions_net_train_rhs):        
        
        if pars['add_repulsion'] in [1,2,3,4,5,6,7,8] and self.itges >= pars['i_start_repulsion']:
            if pars['add_repulsion'] == 1:
                buf0 = [predictions_net_train[j_ens][:,0] for j_ens in range(pars['N_ensemble'])] #buf0 necessary to avoid "backward through graph again"
                buf = torch.cat(predictions_net_train,1).T                    
            if pars['add_repulsion'] == 2:
                buf0 = [net.dpar[:pars['Npar']] for net in self.nets_pinn]
                buf = torch.cat([net.dpar[:pars['Npar']].unsqueeze(0) for net in self.nets_pinn],0)
            if pars['add_repulsion'] in [3,4,5,6,7,8]: 
                buf01 = [predictions_net_train[j_ens][:,0] for j_ens in range(pars['N_ensemble'])]
                if pars['learn_rhs'] == True:
                    buf02 = [predictions_net_train_rhs[j_ens][:,0] for j_ens in range(pars['N_ensemble'])]
                    buf0 = [torch.cat((buf01[j],buf02[j]),0) for j in range(pars['N_ensemble'])]
                else:
                    buf02 = [net.dpar[:pars['Npar']] for net in self.nets_pinn]                
                    buf0 = [torch.cat((buf01[j],buf02[j]),0) for j in range(pars['N_ensemble'])]
                
                buf1 = torch.cat(predictions_net_train,1).T  
                if pars['learn_rhs'] == True:
                    buf2 = torch.cat(predictions_net_train_rhs,1).T  
                else:
                    buf2 = torch.cat([net.dpar[:pars['Npar']].unsqueeze(1) for net in self.nets_pinn],1).T  
                buf = torch.cat((buf1, buf2), 1)
             
            losses_repulsion = []
            if pars['repulsive_approximation'] == 'KDE':
                for j_ens in range(pars['N_ensemble']):
                    if pars['add_repulsion'] in [1,2,3]:
                        h_ges = get_h(buf.detach(), opt='factorized')
                        self.h1 = h_ges[0]
                        self.h2 = h_ges[-1]
                        l_buf = torch.log(RBF_kernel(buf0[j_ens].unsqueeze(0), buf[:,:].detach(), h_ges).sum())
                        
                    if pars['add_repulsion'] == 4:
                        h_ges = get_h(buf.detach(), opt='factorized')
                        self.h1 = h_ges[0]
                        self.h2 = h_ges[-1]
                        l_buf = torch.log(RBF_kernel(buf0[j_ens].unsqueeze(0), buf[:,:].detach(), h_ges).sum())
                        
                    if pars['add_repulsion'] == 5:
                        ibuf = predictions_net_train[0].shape[0]
                        h1_ges = get_h(buf[:,:ibuf].detach(), opt='factorized')
                        h2_ges = get_h(buf[:,ibuf:].detach(), opt='factorized')
                        self.h1 = h1_ges[0]
                        self.h2 = h2_ges[0]
                        l_buf0 = RBF_kernel(buf01[j_ens].unsqueeze(0), buf[:,:ibuf].detach(), h1_ges).sum()                  
                        l_buf1 = RBF_kernel(buf02[j_ens].unsqueeze(0), buf[:,ibuf:].detach(), h2_ges).sum()
                        l_buf = torch.log(l_buf0) + torch.log(l_buf1)
                        
                        
                    if pars['add_repulsion'] == 6:                        
                        hs = get_h(buf.detach(), opt='factorized')
                        self.h1 = hs.mean()
                        self.h2 = hs.std()
                        bufs = RBF_kernel(buf0[j_ens].unsqueeze(0), buf.detach(), hs, opt='factorized').sum(0)
                        l_buf = torch.log(bufs).sum()
                        
                    if pars['add_repulsion'] in [7,8]:
                        ibuf = predictions_net_train[0].shape[0]
                        hs = get_h(buf[:,:ibuf].detach(), opt='factorized')
                        self.h1 = hs.mean()
                        bufs = RBF_kernel(buf01[j_ens].unsqueeze(0), buf[:,:ibuf].detach(), hs, opt='factorized').sum(0)
                        l_buf = torch.log(bufs).sum()
                        
                        if pars['add_repulsion'] == 7:
                            h2s = get_h(buf[:,ibuf:].detach(), opt='factorized')   
                            self.h2 = h2s[0]
                            l_buf1 = RBF_kernel(buf02[j_ens].unsqueeze(0), buf[:,ibuf:].detach(), h2s).sum()
                            l_buf = l_buf + torch.log(l_buf1)
                            
                        
                        
                    losses_repulsion.append(l_buf)
                    
                    
            elif pars['repulsive_approximation'] == 'SSGE': #currently not working
                if pars['add_repulsion'] in [1,2,3]:
                    buf1 = torch.vstack(buf0)
                    SSGE = SSGE_estimator(buf1.detach(), device=buf1.device)
                    
                    self.h1 = SSGE.h
                    buf2 = SSGE.estimator(buf1.detach())
                    for j_ens in range(pars['N_ensemble']):
                        losses_repulsion.append(buf0[j_ens]@buf2[j_ens])
                    
                    
        else:
            losses_repulsion = [torch.tensor(0.,device = pars['device']) for j in range(pars['N_ensemble'])]
            
        return losses_repulsion



    def calculate_ensemble_losses(self, ds, t_batch_ges, t_train_batch, y_train_batch, pars):
        losses_ges, losses_pde, losses_data, losses_prior, losses_repulsion, predictions_net_train, predictions_net_train_rhs = [[], [], [], [], [], [], []]
        weights_net_pinn, weights_net_rhs = [[], []]
        for j_ens in range(pars['N_ensemble']):
            loss, loss_f0, loss_d0, loss_prior, y_net_train, y_net_train_rhs = self.calculate_losses(ds, t_batch_ges[j_ens], t_train_batch, y_train_batch, pars, j_ens)
            losses_ges.append(loss)
            losses_pde.append(loss_f0)
            losses_data.append(loss_d0)
            losses_prior.append(loss_prior)
            predictions_net_train.append(y_net_train.reshape(-1).unsqueeze(1))  
            if pars['learn_rhs'] == True:
                predictions_net_train_rhs.append(y_net_train_rhs.reshape(-1).unsqueeze(1))  
            
            if pars['repulsion_space'] == 'w':
                wpinn_buf, wrhs_buf = self.get_weight_list(j_ens)
                weights_net_pinn.append(wpinn_buf.unsqueeze(1))
                if pars['learn_rhs'] == True:
                    weights_net_rhs.append(wrhs_buf.unsqueeze(1))
            
        if pars['repulsion_space'] == 'w':
            losses_repulsion = self.get_losses_repulsion(pars, weights_net_pinn, weights_net_rhs)        
        elif pars['repulsion_space'] == 'f':
            losses_repulsion = self.get_losses_repulsion(pars, predictions_net_train, predictions_net_train_rhs)
            
        for j_ens in range(pars['N_ensemble']):
            Nd = y_train_batch.numel()
            Nd_rhs = y_net_train_rhs.numel() if pars['learn_rhs'] else 0
            rfac = 1/np.sqrt(pars['N_train']*pars['output_dim']) if pars['add_repulsion'] in [6, 7, 8] else 1                
            losses_ges[j_ens] += rfac*pars['lr_fac']/(Nd + Nd_rhs)*losses_repulsion[j_ens]
            
        return losses_ges, losses_pde, losses_data, losses_prior, losses_repulsion, predictions_net_train
    
    
    def get_weight_list(self, j_ens):
        buf = list(self.nets_pinn[j_ens].parameters())
        buf2 = [el.reshape(-1) for el in buf]            
        weights_net_pinn = torch.cat(buf2[1:], 0)
        
        if self.pars['learn_rhs'] == True:                
            buf = list(self.nets_rhs[j_ens].parameters())
            buf2 = [el.reshape(-1) for el in buf]
            weights_net_rhs = torch.cat(buf2[1:], 0)
        else:
            weights_net_rhs = torch.tensor([], device = weights_net_pinn.device)
        
        return weights_net_pinn, weights_net_rhs
    
    def get_dpar_list(self, j_ens):
        dpars = self.nets_pinn[j_ens].dpar[:self.pars['Npar']]
        return dpars
        
    
    def update_pde_weighting(self):
        self.pars['lf_fac_l'] = self.pars['lf_fac']
        if self.pars['lf_schedule'] == 1:
            if self.pars['x_opt'] == 1:
                if self.itges > int(0.75*self.pars['Npinn']):
                    self.pars['lf_fac_l'] = 2*self.pars['lf_fac']        
                if self.itges > int(0.9*self.pars['Npinn']):
                    self.pars['lf_fac_l'] = 5*self.pars['lf_fac']
            if self.pars['x_opt'] == 5:                
                if self.itges > 5000:
                    self.pars['lf_fac_l'] = 5*self.pars['lf_fac']                
                if self.itges > 7500:
                    self.pars['lf_fac_l'] = 10*self.pars['lf_fac']                               
                if self.itges > 9000 and self.pars['learn_pde_pars'] == 0:
                    self.pars['lf_fac_l'] = 20*self.pars['lf_fac']                               
                if self.itges > 12000 and self.pars['learn_pde_pars'] == 0:
                    self.pars['lf_fac_l'] = 50*self.pars['lf_fac']
            if self.pars['x_opt'] == 7:
                if self.itges > 5000:
                    self.pars['lf_fac_l'] = 2*self.pars['lf_fac']                
                if self.itges > 7500:
                    self.pars['lf_fac_l'] = 4*self.pars['lf_fac']
                    
      
    def Gaussian_log_prob(self, y, mean = 0, sigma=1):
        return -(y - mean)**2/(2*sigma**2) - 0.5*torch.log(2.*sigma**2*torch.pi)
      
      
    def calculate_losses_prior(self, t_net, y_net, j_ens):
        loss_prior = 0
        if 'w' in self.pars['priors']:
            weights_net_pinn, weights_net_rhs = self.get_weight_list(j_ens)
            weights = torch.cat((weights_net_pinn, weights_net_rhs),0)
            prior_mean, prior_std = torch.tensor([0., self.pars['sigma_w']], device=t_net.device)
            loss_prior += -self.Gaussian_log_prob(weights, prior_mean, prior_std).sum()
        if 'f' in self.pars['priors']:
            loss_prior += -100*torch.min(torch.tensor(0.), y_net.view(-1)).sum() #~ uniform prior on positive half-axis
        if 'l' in self.pars['priors']:
            if self.pars['x_opt'] == 1:
                dpars = self.get_dpar_list(j_ens)
                prior_mean, prior_std = torch.tensor([0., 1.], device=t_net.device)
                loss_prior += -self.Gaussian_log_prob(dpars, prior_mean, prior_std)
            if self.pars['x_opt'] in [5, 7, 102]:
                dpars = self.get_dpar_list(j_ens)
                loss_prior += -1000*(-torch.min(torch.tensor(0.), dpars)**2).sum() #~log(exp(-dpar)) on negative half-axis; to approximate uniform prior on positive half-axis
        return loss_prior
            
            
            
    def calculate_losses(self, ds, t_batch, t_train_batch, y_train_batch, pars, j_ens):
        t_batch.requires_grad = True
        t_train_batch.requires_grad = True
        t_ges = torch.cat((t_batch, t_train_batch),0)
        y_ges, x_ges = self.calculate_prediction(t_ges, ds, j_ens)     
        lf_ges = self.get_pde_residuals(t_ges, x_ges, j_ens)    
        y_net_train = y_ges[t_batch.shape[0]:]   
        y_net_train_rhs = self.nets_rhs[j_ens](self.data['t_train_rhs']) if self.learn_rhs == True else False
        #residuals = self.get_data_residuals(y_train_batch, y_net_train)            
        
        loss_f0 = lf_ges[:t_batch.shape[0]].mean() #pde loss at the collocation points
        loss_d0 = self.fld0(y_train_batch, y_net_train).mean() #data loss of function values
        loss_drhs = self.fld0(self.data['y_train_rhs'], y_net_train_rhs).mean() if self.learn_rhs == True else 0 #data loss of rhs values
        loss_prior = self.calculate_losses_prior(t_batch, y_ges[:t_batch.shape[0]], j_ens)
        
        
        Nd = y_train_batch.numel()
        Nd_rhs = y_net_train_rhs.numel() if self.learn_rhs == True else 0
        
        ### total loss     
        loss_data = (Nd*loss_d0 + Nd_rhs*loss_drhs)/(Nd + Nd_rhs)
        loss_prior = loss_prior/(Nd + Nd_rhs)
        loss = pars['ld_fac']*loss_data + pars['lr_fac']*loss_prior + pars['lf_fac_l']*(loss_f0)
        
        
        return loss, loss_f0, loss_d0, loss_prior, y_net_train, y_net_train_rhs
        
    
    #%% functions for model evaluation
    
    
    def get_t_trains_etc(self, j_ens=False):
        nets = [self.nets_pinn[j_ens]] if type(j_ens) is int else False
        t_trains = [self.data['t_train'].detach()]
        y_trains = [self.data['y_train'].cpu()]
        t_tests = [self.data['t_test'].detach()]
        y_tests = [self.data['y_test'].cpu()]
        if self.pars['learn_rhs'] == True:
            if type(j_ens) is int:
                nets.append(self.nets_rhs[j_ens])
            t_trains.append(self.data['t_train_rhs'].detach())
            y_trains.append(self.data['y_train_rhs'].cpu())
            t_tests.append(self.data['t_test_rhs'].detach())
            y_tests.append(self.data['y_test_rhs'].cpu())
            
        return nets, t_trains, y_trains, t_tests, y_tests
    
    
    def calculate_ensemble_predictions(self):
        fpred_train = torch.zeros(self.pars['N_ensemble'], *self.data['y_train'].shape)
        fpred_test = torch.zeros(self.pars['N_ensemble'], *self.data['y_test'].shape)    
        fpred_plot = torch.zeros(self.pars['N_ensemble'], *self.data['f_plot'].shape) 
        fpred_plot_rhs = -1
        if self.pars['learn_rhs'] == True:
            fpred_train_rhs = torch.zeros(self.pars['N_ensemble'], *self.data['y_train_rhs'].shape)
            fpred_test_rhs = torch.zeros(self.pars['N_ensemble'], *self.data['y_test_rhs'].shape)
            fpred_plot_rhs = torch.zeros(self.pars['N_ensemble'], *self.data['f_plot_rhs'].shape)    
        
        with torch.no_grad():
            for j_ens in range(self.pars['N_ensemble']):            
                fpred_train[j_ens] = self.nets_pinn[j_ens](self.data['t_train']).detach().cpu()
                fpred_test[j_ens] = self.nets_pinn[j_ens](self.data['t_test']).detach().cpu()
                fpred_plot[j_ens] = self.nets_pinn[j_ens](self.data['t_plot']).detach().cpu()
                if self.pars['learn_rhs'] == True:
                    fpred_train_rhs[j_ens] = self.nets_rhs[j_ens](self.data['t_train_rhs']).detach().cpu()
                    fpred_test_rhs[j_ens] = self.nets_rhs[j_ens](self.data['t_test_rhs']).detach().cpu()
                    fpred_plot_rhs[j_ens] = self.nets_rhs[j_ens](self.data['t_plot']).detach().cpu()
        
        self.fpred_trains = [fpred_train]
        self.fpred_tests = [fpred_test]
        self.fpred_plots = [fpred_plot]
        if self.pars['learn_rhs'] == True:
            self.fpred_trains.append(fpred_train_rhs)
            self.fpred_tests.append(fpred_test_rhs)
            self.fpred_plots.append(fpred_plot_rhs)
            
        return fpred_plot, fpred_plot_rhs


    def calculate_model_evaluations(self, j_ens):
        self.get_model_RMSE(self, j_ens)
        self.get_model_logL(self, j_ens)
    
    def calculate_ensemble_evaluations(self):
        logL_ges = []
        RMSE_ges = []
        for j_ens in range(self.pars['N_ensemble']):
            logL_ges.append(self.calculate_logL(j_ens))
            RMSE_ges.append(self.calculate_RMSE(j_ens))
        
        logL_ensemble = self.calculate_ensemble_logL()
        RMSE_ensemble = self.calculate_ensemble_RMSE()
        return logL_ges, RMSE_ges, logL_ensemble, RMSE_ensemble
        
    def calculate_RMSE(self, j_ens):    
        N = 0
        RMSE_train = 0
        RMSE_test = 0
        
        _, t_trains, y_trains, t_tests, y_tests = self.get_t_trains_etc()        
        for fpred_train, y_train, fpred_test, y_test in zip(self.fpred_trains, y_trains, self.fpred_tests, y_tests):
            N += y_train.shape[0]
            RMSE_train += torch.sum((fpred_train[j_ens].squeeze() - y_train.squeeze())**2)
            RMSE_test += torch.sum((fpred_test[j_ens].squeeze() - y_test.squeeze())**2)
            
        RMSE_train = torch.sqrt(RMSE_train/(self.pars['N_train'] + self.pars['N_train_rhs']))
        RMSE_test = torch.sqrt(RMSE_test/(self.pars['N_test'] + self.pars['N_test_rhs']))
        return [RMSE_train, RMSE_test]

    def calculate_logL(self, j_ens):
        logL_train = 0 
        logL_test = 0         
            
        nets, t_trains, y_trains, t_tests, y_tests = self.get_t_trains_etc(j_ens)            
        for net, t_train, y_train, t_test, y_test in zip(nets, t_trains, y_trains, t_tests, y_tests):
            ypred_train = net(t_train).detach().cpu()
            logL_train += (-(ypred_train - y_train)**2).sum()
            ypred_test = net(t_test).detach().cpu()
            logL_test += (-(ypred_test - y_test)**2).sum()     

             
        return [logL_train, logL_test]
    
    def calculate_ensemble_RMSE(self):
        RMSE_train = 0
        RMSE_test = 0
        RMSE_true = 0
       
        _, t_trains, y_trains, t_tests, y_tests = self.get_t_trains_etc()        
        for fpred_train, y_train, fpred_test, y_test in zip(self.fpred_trains, y_trains, self.fpred_tests, y_tests):
            RMSE_train += ((fpred_train.mean(0).squeeze() - y_train.squeeze(1))**2).sum()
            RMSE_test += ((fpred_test.mean(0).squeeze() - y_test.squeeze(1))**2).sum()
            RMSE_true += ((fpred_test.mean(0).squeeze() - self.data['f_test'].squeeze().cpu())**2).sum()
            
        RMSE_train = torch.sqrt(RMSE_train/(self.pars['N_train'] + self.pars['N_train_rhs']))
        RMSE_test = torch.sqrt(RMSE_test/(self.pars['N_test'] + self.pars['N_test_rhs']))
        RMSE_true = torch.sqrt(RMSE_true/(self.pars['N_test'] + self.pars['N_test_rhs']))
        return [RMSE_train, RMSE_test, RMSE_true]

    def calculate_ensemble_logL(self):
        logL_train = 0
        logL_test = 0
        logL_true = 0
        
        
        if self.pars['N_ensemble'] > 1:
            _, t_trains, y_trains, t_tests, y_tests = self.get_t_trains_etc()            
            for fpred_train, y_train, fpred_test, y_test in zip(self.fpred_trains, y_trains, self.fpred_tests, y_tests):
                for k in range(fpred_train.shape[1]):
                    for j in range(fpred_train.shape[2]):
                        buf = fpred_train[:,k,j]
                        kde = KernelDensity(kernel='gaussian', bandwidth=get_h(buf)).fit(buf.unsqueeze(1))
                        logL_train += kde.score_samples(y_train[k,j].unsqueeze(0).unsqueeze(1)).squeeze()
                        
                for k in range(fpred_test.shape[1]):
                    for j in range(fpred_test.shape[2]):
                        buf = fpred_test[:,k,j]
                        kde = KernelDensity(kernel='gaussian', bandwidth=get_h(buf)).fit(buf.unsqueeze(1))
                        logL_test += kde.score_samples(y_test[k,j].unsqueeze(0).unsqueeze(1)).squeeze()
                        logL_true += kde.score_samples(self.data['f_test'][k,j].cpu().unsqueeze(0).unsqueeze(1)).squeeze()
                        
                
        
        return [logL_train, logL_test, logL_true]

    def evaluate_ensemble(self, jm, res):
        res.xnp_ges, res.xnp_rhs_ges = self.calculate_ensemble_predictions()
        res.xnp_ges = res.xnp_ges.reshape([-1] + list(res.pars['f_plot_shape']))
        if type(res.xnp_rhs_ges) is not int:
            res.xnp_rhs_ges = res.xnp_rhs_ges.reshape([-1] + res.pars['f_plot_shape'])            
        res.logL_ges, res.RMSE_ges, res.logL_ensemble_ges, res.RMSE_ensemble_ges = self.calculate_ensemble_evaluations()
        