# -*- coding: utf-8 -*-
import os
import random
import shutil
import torch
import numpy as np
import pickle
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


#initialize random seeds
def init_random_seeds(s=False):
    if type(s) == bool:
        s = s = np.random.randint(42*10**4)
        
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    
    rand_init = 1
    return rand_init, s

#create string leading to folder; also create folder
def set_input_path(path0, folder0, replace=True):
    input_path = path0 + folder0 + '/'
    check_dirs(path0, input_path, replace=True)    
    print(input_path)
    return input_path

#check if input_path exists and create if it does not
def check_dirs(path0, input_path, replace=False):
    
    if not os.path.exists(input_path):
        os.mkdir(input_path)    
    if not os.path.exists(input_path + 'input.py') or replace == True:
        shutil.copyfile(path0 + 'input.py', input_path + 'input.py')
        

def initialize_writer(log_dir, comment0 = "", comment = ""):
    #comment=""
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, comment0 + "_" + current_time + "_" + comment)
    return SummaryWriter(log_dir = log_dir)

        
def init_par(pars, key, val):
    if not key in pars: pars[key] = val
    
#update parameter value to new value
def update_par(pars, key, val, it=None, rpath=None):
    pars[key] = val
    if not it is None:
        print_parameter_update(key, val, it, rpath)
        
#write parameter update to readme
def print_parameter_update(key, val, it, rpath):
    freadme = rpath + '/../readme.txt'
    with open(freadme, 'a+') as f:
        f.write('it' + str(it) + '_' + key + ':' + str(val) + '\n')
        
#print parameters
def print_parameters(pars, fpath):
    with open(fpath, 'w') as f:
        for var in sorted(pars.keys(), key=str.casefold):
            f.write(var+':'+str(pars[var])+'\n')
        f.write('\n\n')


#keep track of parameter values before choosing model
def add_to_initial(pars, res, key):
    res.initial_pars[key] = pars[key]
    
#reset parameter values to initial values
def reset_to_initial(pars, res):
    for key in res.initial_pars:
        update_par(pars, key, res.initial_pars[key])

#write to same line in console
def write_and_flush(msg):
    sys.stdout.write('\r'+msg)
    sys.stdout.flush()
    
def switch_device(vvec, device = 'cpu'):
    for v in vvec:
        v.to(device)

        
    
#define a new model given keys and corresponding vals
def define_model(keys, vals, res, pars, it=None, rpath=None):
    mname = ''
    for j in range(len(keys)):
        if j > 0 :
            mname = mname + '-'
        key = keys[j]
        val = vals[j]
        add_to_initial(pars, res, key)
        update_par(pars, key, val)
        mname = mname + key + ' - ' + str(val)
    return mname

def get_ds_name(x_opt):
    if x_opt == 1:
        return 'exp'
    if x_opt == 2:
        return 'sin'
    if x_opt == 3:
        return 'Bessel'
    if x_opt == 4:
        return 'sin2d'
    if x_opt == 5:
        return 'damped_oscillator'
    if x_opt == 7:
        return 'hare_lynx'
    if x_opt == 51:
        return 'test'
    if x_opt == 52:
        return 'test_wilson'
    if x_opt == 101:
        return 'NS'
    if x_opt == 102:
        return 'advection'
    if x_opt == 103:
        return 'Burgers'

def get_mname(jm, rep_space = 'fl'):
    return get_lvec(rep_space = rep_space)[jm]
    
        
    
#set parameter values to those corresponding to chosen model
def update_model_pars(jm, res, pars, rpath='', it=0):
    pars['jm'] = jm
    if jm == 0:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [0],
            res, pars)
    if jm == 1:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [1],
            res, pars)
    if jm == 2:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [2],
            res, pars)
    if jm == 3:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [3],
            res, pars)
    if jm == 4:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [4],
            res, pars)
    if jm == 5:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [5],
            res, pars)
    if jm == 6:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [6],
            res, pars)
    if jm == 7:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [7],
            res, pars)
    if jm == 8:
        mname = get_mname(jm)
        define_model(
            ['add_repulsion'],
            [8],
            res, pars)
        
    res.mname_vec[jm] = mname
    
    

#initialize parameters and bookkeeping devices for current run of model
def initialize_model_parameters(res, pars, comment, jN, jm):
    reset_to_initial(pars, res) #reset parameters to model-independent values
    update_model_pars(jm, res, pars) #update model specific parameters
    fpath = res.rpath + '/../m'+str(jm)+'_pars.txt'
    print_parameters(pars, fpath) #print parameters to text file

    writer = initialize_writer("../runs/runs_punc/", comment0 = 'ds' + str(pars['x_opt']) + 'm'+str(jm) + '_' + pars['collocation_opt'], comment = comment)
    print_parameters(pars, writer.log_dir + '/pars.txt')
    
    return writer, fpath
    
    
    

        
sname_vec = ['logL validation','RMSE validation',r'$f^2$', r'$|\Delta \lambda|$']      
#%%%%
#get names of models
def get_lvec(rep_space='fl', opt='', opt2=''):
    if rep_space == 'fl':
        r1 = 'f'
        r2 = r'$\lambda$'
    if rep_space == 'fg':
        r1 = 'f'
        r2 = 'g'
    if rep_space == 'wl':
        r1 = r'$\theta$'
        r2 = r'$\lambda$'
    if rep_space == 'ww':
        r1 = r'$\theta$'
        r2 = r"$\theta'$"
        
    if opt == 'table':
        if opt2 == '':
            return ['non-repulsive',
                    'repulsive ('+r1+')', 
                    'repulsive ('+r2+')',
                    'repulsive ('+r1+' and '+r2+') \\  (same h)}', 
                    'repulsive ('+r1+' and '+r2+r')', 
                    r'\makecell{repulsive ('+r1+' and '+r2+r') \\ (factorized)}', 
                    r'\makecell{repulsive ('+r1+' and '+r2+r') \\ (fully factorized)}',
                    r'\makecell{repulsive ('+r1+' and '+r2+r') \\ ('+r1+' factorized)}',
                    r'\makecell{repulsive ('+r1+r') \\ (factorized)}'
                    ]
        elif opt2 == 'flipped':
            return ['non-repulsive',
                    'repulsive ('+r1+')', 
                    'repulsive ('+r2+')',
                    'repulsive ('+r1+' and '+r2+') (same h)}', 
                    'repulsive ('+r1+' and '+r2+r')', 
                    'repulsive ('+r1+' and '+r2+r') (factorized)', 
                    'repulsive ('+r1+' and '+r2+r') (fully factorized)',
                    'repulsive ('+r1+' and '+r2+r') ('+r1+' factorized)',
                    'repulsive ('+r1+r') (factorized)'
                    ]
    else:
        return ['non-repulsive',
                'repulsive ('+r1+')', 
                'repulsive ('+r2+')',
                'repulsive ('+r1+' and '+r2+') \n  (same h)', 
                'repulsive ('+r1+' and '+r2+')', 
                'repulsive ('+r1+' and '+r2+') \n (factorized)', 
                'repulsive ('+r1+' and '+r2+') \n (fully factorized)',
                'repulsive ('+r1+' and '+r2+') \n ('+r1+' factorized)',
                'repulsive ('+r1+') \n (factorized)']

        
def get_rep_space(pars):
    if pars['repulsion_space'] == 'f' and pars['learn_rhs'] == 0:
        rep_space = 'fl'
    if pars['repulsion_space'] == 'f' and pars['learn_rhs'] == 1:
        rep_space = 'fg'
    if pars['repulsion_space'] == 'w' and pars['learn_rhs'] == 0:
        rep_space = 'wl'
    if pars['repulsion_space'] == 'w' and pars['learn_rhs'] == 1:
        rep_space = 'ww'
    return rep_space

def get_par_name(pars, jl):
    if pars['x_opt'] == 7:
        if jl == 0:
            return r'$\alpha$'
        if jl == 1:
            return r'$\beta$'
        if jl == 2:
            return r'$\gamma$'
        if jl == 3:
            return r'$\delta$'
    elif pars['x_opt'] ==  5:
        if jl == 0:
            return r'$\omega$'
        if jl == 1:
            return r'$\gamma$'
    else:
        if len(pars['dpar']) > 1:
            return r'$\lambda_{%i}$'%(jl)
        else:
            return r'$\lambda$'


#get name of jm-th model
def get_label(jm, rep_space='fl', opt='', opt2=''):
    lvec = get_lvec(rep_space=rep_space, opt=opt, opt2=opt2)
    if jm < len(lvec) and jm >= 0:
        return lvec[jm]
    else:
        return 'Monte Carlo'

#get color for jm-th model
def get_color(jm):
    cvec = ['k','r','y','b','m','c','g','C0','C2']  
    return cvec[jm]


