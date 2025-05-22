# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy as sp
from torchdiffeq import odeint

from .utils import init_par

def get_ds(pars):
    x_opt = pars['x_opt']
    if x_opt == 1:
        ds = ds_exp_1d(pars)
    if x_opt == 5:
        ds = ds_damped_oscillator(pars)
    if x_opt == 7:
        ds = ds_lotka_volterra(pars)
    if x_opt == 52:
        ds = ds_dummy(pars)
    if x_opt == 102:
        ds = ds_advection_2d(pars)
        
    return ds



class pebm_dataset():
    def __init__(self, pars):
        self.x_opt = pars['x_opt']
        self.dpar = pars['dpar']
        
        self.init_train_pars(pars)
        self.init_data_ranges(pars)
        self.init_network_pars(pars)
        
        pars['nfac'] = pars['nfac0']*pars['fnoise']
        pars['dim'] = len(pars['tmin'])        
        
    def init_data_ranges(self, pars):  #ranges of input data and normalizing const.
        print('not implemented!')
    
    def init_network_pars(self, pars):  #network parameteres        
        init_par(pars, 'Uvec_pinn',  [40]*4)
        init_par(pars, 'Uvec_rhs', [20]*2)
        init_par(pars, 'fdrop_pinn', 0.)
        init_par(pars, 'Uvec_ebm', [5]*3)
        init_par(pars, 'fdrop_ebm', 0.5)
        init_par(pars, 'output_dim', 1)
    
    def init_train_pars(self, pars):  #training parameters
    
        init_par(pars, 'bs_coll', 100)
        init_par(pars, 'bs_train', 200)
        init_par(pars, 'lr_pinn', 2e-3)
        init_par(pars, 'lr_ebm', 2e-3)
        init_par(pars, 'ld_fac', 1)
        init_par(pars, 'lf_fac', 1)
        init_par(pars, 'Nebm', 2000)
        init_par(pars, 'i_init_ebm', -1)
        init_par(pars, 'Npinn', 3000)
        init_par(pars, 'Npar', len(pars['dpar']))
        
        
    def get_fx0(self, nu):  #definition of function
        print('not implemented!')
        return -1
    
    def get_fx(self, nu=None):  #definition of function
        if nu is None:
            nu = self.dpar
        if type(nu) is not list:
            nu = [nu]
        return self.get_fx0(nu)
        
    def get_frhs(self): #definition of functional form of rhs
        print('not implemented!')
        return -1
    
    def get_loss_d(self, n_opt = 'G0'):
        def loss_d(y_true, y_net):
            return (y_true.squeeze() - y_net.squeeze())**2
        return loss_d
    
    
    def get_loss_f(self):  #definition of pde loss
        print('not implemented!')        
        
    def set_temp(self, t, x, dpar):
        return -1
        
    def get_meas_eq(self, y_opt=-1):
        def meas_eq(x):
            if y_opt == -1:
                return x
        return meas_eq

class ds_dummy(pebm_dataset):
    def __init__(self, pars):
        init_par(pars, 'dpar', [1])
        init_par(pars, 'Uvec_pinn',  [50]*2)
        pebm_dataset.__init__(self, pars)
        
    def init_data_ranges(self, pars):
        
        init_par(pars, 'tmin', (-10,))        
        init_par(pars, 'tmax', (10,))        
        init_par(pars, 'tmin_coll', (-20,))      
        init_par(pars, 'tmax_coll', (20,))
        init_par(pars, 'ylim', [-0.5, 5])
        init_par(pars, 'nfac0', 0.001)
        init_par(pars, 'N_train', 200)        
        init_par(pars, 'N_coll', 2000)  
        init_par(pars, 'N_test', 20)   
        
    def get_fx(self):
        return lambda t: 0*t
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            lf = 0*t**2
            return lf
        return loss_f


 
        
class ds_exp_1d(pebm_dataset):
    def __init__(self, pars):
        init_par(pars, 'dpar', [0.3])
        pebm_dataset.__init__(self, pars)
        
    def init_data_ranges(self, pars):
        init_par(pars, 'Uvec_pinn',  [20]*2)
        init_par(pars, 'tmin', (0,))        
        init_par(pars, 'tmax', (10,))        
        init_par(pars, 'tmin_coll', (0,))      
        init_par(pars, 'tmax_coll', (15,))
        init_par(pars, 'ylim', [-2, 30])
        init_par(pars, 'nfac0', 1)
        init_par(pars, 'N_train', 200)        
        init_par(pars, 'N_coll', 2000)  
        init_par(pars, 'N_test', 15)   
        
        init_par(pars, 'tmin_rhs', (0,))
        init_par(pars, 'tmax_rhs', (15,))
        init_par(pars, 'N_train_rhs', 15)
        init_par(pars, 'N_test_rhs', 15)   
    
    def get_fx(self):
        return lambda t: torch.exp(self.dpar[0]*t)
    
    def get_frhs(self):
        return lambda t: self.dpar[0]*torch.exp(self.dpar[0]*t)
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            lf = (xdot - dpar[0]*x)**2
            return lf
        return loss_f


    
    

    
    
class ds_damped_oscillator(pebm_dataset):
    def __init__(self, pars):
        init_par(pars, 'dpar', [1.0, 0.1])  # dpar = [omega, zeta]
        init_par(pars, 'Uvec_pinn', [20]*3)
        pebm_dataset.__init__(self, pars)

    def init_data_ranges(self, pars):
        init_par(pars, 'tmin', (0,))
        init_par(pars, 'tmax', (20,))
        init_par(pars, 'tmin_coll', (0,))
        init_par(pars, 'tmax_coll', (25,))
        init_par(pars, 'ylim', [-1.5, 1.5])
        init_par(pars, 'nfac0', 0.1)
        init_par(pars, 'N_train', 200)
        init_par(pars, 'N_coll', 2000)
        init_par(pars, 'N_test', 15)

        init_par(pars, 'tmin_rhs', (0,))
        init_par(pars, 'tmax_rhs', (20,))
        init_par(pars, 'N_train_rhs', 30)
        init_par(pars, 'N_test_rhs', 15)

    def get_fx(self):
        omega, zeta = torch.tensor(self.dpar)
        return lambda t: torch.exp(-zeta * omega * t) * torch.sin(omega * torch.sqrt(1 - zeta**2) * t)

    def get_frhs(self):
        return lambda t: 0.0 * t  # or just return None

    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot2 = torch.autograd.grad(xdot.sum(), t, create_graph=True)[0]

            omega, zeta = dpar[:2]
            lf = (xdot2 + 2 * zeta * omega * xdot + omega**2 * x) ** 2

            return lf
        return loss_f
    


    
    
		
class ds_advection_2d(pebm_dataset):
    def __init__(self, pars):
        init_par(pars, 'dpar', [0.2])
        pebm_dataset.__init__(self, pars)
        
    def init_data_ranges(self, pars):   
        init_par(pars, 'Uvec_pinn',  [100]*4)
        init_par(pars, 'tmin', (0,0))
        init_par(pars, 'tmax', (2,1))        
        init_par(pars, 'tmin_coll', (0,0))
        init_par(pars, 'tmax_coll', (2,1))
        init_par(pars, 'nfac0', 0.1)#03)
        init_par(pars, 'N_train', 200)        
        init_par(pars, 'N_coll', 2000)
    
        
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot0 = xdot[:,0]
            xdot1 = xdot[:,1]            
            lf = (xdot0 + dpar[0]*xdot1)**2
            return lf
        return loss_f

    
    
  
    
############################################
############################################
############################################
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



class ds_lotka_volterra(pebm_dataset):
    def __init__(self, pars):
        #init_par(pars, 'dpar', [1., 0.02, 2.0, 0.025])   
        init_par(pars, 'dpar', [1.5, 1., 3., 1.])   
        self.y0 = [10.0, 5.0]
        init_par(pars, 'y0', self.y0)
        #init_par(pars, 'dpar', [1., 0.1, 1.1, 0.075])       
        pars['output_dim'] =  2
        self.ODE = LotkaVolterraODE(*pars['dpar'])
        pebm_dataset.__init__(self, pars)
    
    def init_network_pars(self, pars):  #network parameters
        init_par(pars, 'Uvec_pinn',  [20]*3)
        init_par(pars, 'fdrop_pinn', 0.)
        
    def init_data_ranges(self, pars): 
        init_par(pars, 'tmin', (0.,))
        init_par(pars, 'tmax', (10.,))        
        init_par(pars, 'tmin_coll', (0.,))
        init_par(pars, 'tmax_coll', (10.,))     
        init_par(pars, 'ylim', [-1., 15.])
        init_par(pars, 'nfac0', 1.0)
        init_par(pars, 'N_train', 15)   
        init_par(pars, 'N_test', 7)
        init_par(pars, 'N_coll', 100)
        
    
    def get_fx(self):
        return lambda t: odeint(self.ODE, torch.tensor(self.y0), t)
        
        
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            x = x.squeeze()
            xdot0 = torch.autograd.grad(x[:,0].sum(), t, create_graph=True)[0].squeeze()
            xdot1 = torch.autograd.grad(x[:,1].sum(), t, create_graph=True)[0].squeeze()
            lf0 = (xdot0 - x[:,0]*(dpar[0] - dpar[1]*x[:,1]))**2 
            lf1 = (xdot1 + x[:,1]*(dpar[2] - dpar[3]*x[:,0]))**2      
            lf = lf0 + lf1            
            
            return lf
        return loss_f


