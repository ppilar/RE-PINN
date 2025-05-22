# -*- coding: utf-8 -*-
import numpy as np
import torch
import random

from scipy.stats import skewnorm
from scipy.special import erf

def get_noise(n_opt, f, pars=0):
    #load desired noise function
    
    if n_opt == 'G': #Gaussian
        return n_G(n_opt, f, pars)
    print('error! noise not defined!')
    
    
class Noise():
    def __init__(self, n_opt, f, pars=[]):
        self.n_opt = n_opt
        self.f = f
        self.pars = pars
        self.mu = self.mu0*self.f
        self.sig = self.sig0*self.f
        
    def sample(self, Ns):
        #allows to sample from noise distribution
        print('not implemented!')
        
    def pdf(self, x):
        #returns the noise pdf(x)
        print('not implemented!')
        
    def init_pars(self):
        #initializes the noise distribution either with the argument pars, if given, or with standard values
        print('not implemented!')

class n_G(Noise):
    #Gaussian noise
    
    def __init__(self, n_opt, f, pars):
        if type(pars) == int: pars = self.init_pars(pars)        
        self.mu0, self.sig0 = pars
        Noise.__init__(self, n_opt, f, pars)
        
    def sample(self, Ns):
        return torch.tensor(np.random.normal(self.mu0*self.f, self.sig0*self.f, Ns))
    
    def pdf(self, x):
        q = torch.distributions.normal.Normal(self.mu0*self.f,self.sig0*self.f)
        pdf = torch.exp(q.log_prob(x))
        return pdf
    
    def init_pars(self, pars):
        pars = [0,1]
        return pars
    

        
        
        