# -*- coding: utf-8 -*-

import time
import torch
import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d, RegularGridInterpolator

from .utils import init_par, get_mname, get_ds_name, check_dirs
from .utils_repulsion import get_h

#class to store results
class Results():
    def __init__(self, pars, input_path):
        self.pars = pars
        self.rpath, self.ppath0 = self.create_dirs(input_path, get_ds_name(pars['x_opt']))
        self.initial_pars = dict()
        self.mname_vec = [get_mname(jm) for jm in range(pars['Nmodel'])]
        self.isamples = []
        
        
        #store training results
        self.dpar_gesges = np.zeros([pars['Nmodel'], pars['Npar'], pars['N_ensemble'], pars['Nrun'], pars['Npinn']])
        self.loss_ges_gesges, self.loss_data_gesges, self.loss_pde_gesges, self.loss_rep_gesges = np.zeros([4, pars['Nmodel'], pars['N_ensemble'],  pars['Nrun'], pars['Npinn']])
        self.rmse_data_gesges, self.rmse_pde_gesges = np.zeros([2, pars['Nmodel'], 2, pars['N_ensemble'], pars['Nrun'], pars['Npinn']//pars['itest']])
        self.t_model_gesges = np.zeros([pars['Nmodel'], pars['N_ensemble'], pars['Nrun']])

        
        self.logL_gesges = np.zeros([pars['Nmodel'], pars['N_ensemble'],  pars['Nrun'], 2])
        self.logL_ensemble_gesges = np.zeros([pars['Nmodel'],  pars['Nrun'], 3])
        self.RMSE_gesges = np.zeros([pars['Nmodel'], pars['N_ensemble'],  pars['Nrun'], 2])
        self.RMSE_ensemble_gesges = np.zeros([pars['Nmodel'],  pars['Nrun'], 3])
        
        self.data = []
        
       
        
    def create_dirs(self, input_path, ds_name):
        dpath = input_path + ds_name + '/'
        rpath = dpath + 'files/'
        ppath0 = dpath + 'plots/'        
        
        check_dirs(input_path, dpath)    
        check_dirs(input_path, ppath0)
        check_dirs(input_path, rpath)

        return rpath, ppath0
    
    def init_plot_arrays(self):
        self.xnp_gesges, self.xnp_rhs_gesges = np.zeros([2, self.pars['Nmodel'], self.pars['N_ensemble'],  self.pars['Nrun']] + list(self.pars['f_plot_shape']))
        
        
        
    #initialize empty lists and arrays for storing results of current run
    def init_run_results(self):     
        self.dpar_ges = tuple([[[] for j in range(self.pars['Npar'])] for i in range(self.pars['Nmodel'])])
        self.loss_data_ges, self.loss_pde_ges, self.loss_ges_ges, self.loss_rep_ges = [[[] for k in range(self.pars['Nmodel'])] for j in range(4)]  
        self.rmse_data_ges, self.rmse_pde_ges = [[([],[]) for k in range(self.pars['Nmodel'])] for j in range(2)]      
        self.t_model_ges = np.zeros([self.pars['Nmodel']])
                
        self.xnp_ges = [[] for j in range(self.pars['Nmodel'])]
        self.xnp_rhs_ges = [[] for j in range(self.pars['Nmodel'])]
        self.logL_ges = [[] for j in range(self.pars['Nmodel'])]
        self.logL_ensemble_ges = [[] for j in range(self.pars['Nmodel'])]
        self.RMSE_ges = [[] for j in range(self.pars['Nmodel'])]
        self.RMSE_ensemble_ges = [[] for j in range(self.pars['Nmodel'])]
        self.logL_dpar_ensemble_gesges = np.zeros([self.pars['Nmodel'],  self.pars['Nrun']])
        self.ABS_dpar_ensemble_gesges = np.zeros([self.pars['Nmodel'],  self.pars['Nrun']])
        
        
        self.isamples.append(self.pars['isample'])
        

    
    def calculate_res_stats(self, jN):                
        for jm in self.pars['model_vec']:
            # evaluate function
            
            if self.pars['learn_rhs']:
                raise NotImplementedError()
                
            #interpolate, when necessary
            if self.pars['dim'] == 1:
                interp_func = interp1d(self.data[jN]['t_plot'][:,0], self.xnp_gesges[jm, :, jN, :, :], axis=1, kind='cubic')
                fpred_train = interp_func(self.data[jN]['t_train'][:,0])
                fpred_test = interp_func(self.data[jN]['t_test'][:,0])            
                fpreds = [fpred_train, fpred_test]
            elif self.pars['dim'] == 2:
                tbuf = self.data[jN]['t_plot'].reshape(*self.pars['t_plot_shape'], 2)
                t0 = tbuf[:,0,0]
                t1 = tbuf[0,:,1]
                fpred_train, fpred_test = [], []
                for j_ens in range(self.pars['N_ensemble']):
                    interp_func = RegularGridInterpolator((t0, t1), self.xnp_gesges[jm, j_ens, jN, :, :],
                                     bounds_error=False, fill_value=None)
                    fpred_train.append(interp_func(self.data[jN]['t_train']))
                    fpred_test.append(interp_func(self.data[jN]['t_test']))          
                
                fpred_train = np.stack(fpred_train, 0)
                fpred_test = np.stack(fpred_test, 0)
                fpreds = [fpred_train, fpred_test]
            
            #calculate logL on train, test, and true data
            self.logL_ensemble_gesges[jm, jN, :] *= 0
            for jpred, fpred in enumerate(fpreds):
                for k in range(fpred.shape[1]):
                    for j in range(fpred.shape[2]):
                        buf = torch.tensor(fpred[:,k,j])
                        kde = KernelDensity(kernel='gaussian', bandwidth=get_h(buf)).fit(buf.unsqueeze(1))
                        
                        if jpred == 0:
                            self.logL_ensemble_gesges[jm, jN, 0] += kde.score_samples(self.data[jN]['y_train'][k,j][np.newaxis, np.newaxis]).squeeze()
                        else:
                            self.logL_ensemble_gesges[jm, jN, 1] += kde.score_samples(self.data[jN]['y_test'][k,j][np.newaxis, np.newaxis]).squeeze()
                            self.logL_ensemble_gesges[jm, jN, 2] += kde.score_samples(self.data[jN]['f_test'][k,j][np.newaxis, np.newaxis]).squeeze()
                            
                            
            #calculate RMSE on train, test, and true data       
            self.RMSE_ensemble_gesges[jm, jN, 0] = np.sqrt(((fpred_train.mean(0).squeeze() - self.data[jN]['y_train'].squeeze())**2).sum()/(self.pars['N_train']*self.pars['output_dim']))
            self.RMSE_ensemble_gesges[jm, jN, 1] = np.sqrt(((fpred_test.mean(0).squeeze() - self.data[jN]['y_test'].squeeze())**2).sum()/(self.pars['N_test']*self.pars['output_dim']))
            self.RMSE_ensemble_gesges[jm, jN, 2] = np.sqrt(((fpred_test.mean(0).squeeze() - self.data[jN]['f_test'].squeeze())**2).sum()/(self.pars['N_test']*self.pars['output_dim'])) #TODO: dims
                
            
            
            
            # evaluate lambda 
            logL_dpar_v0 = 0
            self.ABS_dpar_ensemble_gesges[jm,jN] *= 0
            self.logL_dpar_ensemble_gesges[jm,jN] *= 0
            for jl in range(len(self.pars['dpar'])):
                buf_dpar = torch.tensor(self.dpar_gesges[jm,jl,:,jN,-1]).float()                
                self.ABS_dpar_ensemble_gesges[jm,jN] += ((self.pars['dpar'][jl] - buf_dpar.mean()).abs().numpy())/len(self.pars['dpar'])
                kde = KernelDensity(kernel='gaussian', bandwidth=get_h(buf_dpar)).fit(buf_dpar.unsqueeze(1))
                self.logL_dpar_ensemble_gesges[jm,jN] += kde.score_samples(torch.tensor(self.pars['dpar'])[jl].unsqueeze(0).unsqueeze(1)).squeeze()
                
                dpar_dist = torch.distributions.Normal(buf_dpar.mean(), buf_dpar.std())
                logL_dpar_v0 += (dpar_dist.log_prob(torch.tensor(self.pars['dpar'][jl])).item())/len(self.pars['dpar'])
            #print('logL l:', logL_dpar_v0, self.logL_dpar_ensemble_gesges[jm,jN])

    #store train/test data of current run
    def store_run_data(self, model):
        self.data.append({})
        buf = self.data[-1]
        for key in model.data:
            buf[key] = model.data[key]
            if type(model.data[key]) == torch.Tensor:
                buf[key] = buf[key].cpu().numpy()
            

    #store results of current run
    def store_run_results(self, jm, jN):
        it = len(self.dpar_ges[jm][0])
        it_test = len(self.rmse_data_ges[jm][0])
        for jp in range(self.pars['Npar']):     
            self.dpar_gesges[jm,jp,:,jN,:it] = np.array(self.dpar_ges[jm][jp][:]).T   
            
        self.loss_ges_gesges[jm,:,jN,:it] = np.array(self.loss_ges_ges[jm]).T
        self.loss_data_gesges[jm,:,jN,:it] = np.array(self.loss_data_ges[jm]).T
        self.loss_pde_gesges[jm,:,jN,:it] = np.array(self.loss_pde_ges[jm]).T
        if len(self.losses_repulsion) > 0:
            self.loss_rep_gesges[jm,:,jN,:it] = np.array(self.loss_rep_ges[jm]).T
        
        self.rmse_data_gesges[jm,0,:,jN,:it_test] = np.array(self.rmse_data_ges[jm][0]).T #train
        self.rmse_data_gesges[jm,1,:,jN,:it_test] = np.array(self.rmse_data_ges[jm][1]).T #test
        self.rmse_pde_gesges[jm,0,:,jN,:it_test] = np.array(self.rmse_pde_ges[jm][0]).T #train
        self.rmse_pde_gesges[jm,1,:,jN,:it_test] = np.array(self.rmse_pde_ges[jm][1]).T #test
        
        self.t_model_gesges[jm, jN] = self.t_model_ges[jm]
        self.xnp_gesges[jm, :, jN, :] = np.array(self.xnp_ges)
        if self.pars['learn_rhs'] == True:
            self.xnp_rhs_gesges[jm, :, jN, :] = np.array(self.xnp_rhs_ges)
        self.logL_gesges[jm, :, jN, :] = np.array(self.logL_ges)
        self.logL_ensemble_gesges[jm, jN, :] = np.array(self.logL_ensemble_ges)
        self.RMSE_gesges[jm, :, jN, :] = np.array(self.RMSE_ges)
        self.RMSE_ensemble_gesges[jm, jN, :] = np.array(self.RMSE_ensemble_ges)
