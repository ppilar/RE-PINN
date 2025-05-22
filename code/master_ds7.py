# -*- coding: utf-8 -*-

import os
import sys
import random
import torch
import numpy as np
from punc.utils import init_random_seeds, set_input_path, init_par


path0 = '../results/'
iexp_vec = [0]

# if len(sys.argv) > 1:
#     iexp_vec = []
#     for j in range(1, len(sys.argv)):
#         iexp = int(sys.argv[j])
#         iexp_vec.append(iexp)
# else:
#     print('error! no experiments selected!')
print('run experiments:', iexp_vec)



for iexp in iexp_vec:
    print('experiment:', iexp)
    pars = dict()
    init_par(pars, 'model_vec', [0,1,2,4,5,6])
    init_par(pars, 'Nrun', 5)   
    init_par(pars, 'N_ensemble', 25)   
    init_par(pars, 'Npinn', 15000)
    init_par(pars, 'fnoise', 1.)
    init_par(pars, 'x_opt', 7)
    init_par(pars, 'learn_rhs', 0)
    init_par(pars, 'lf_fac', 0.5)
    init_par(pars, 'priors', ['l'])
    
    
    
    comment0 = ''            
    if iexp == 0:
        init_par(pars, 'par_label', 'N_ensemble')
        par_vec = [25]
        comment0 = '_fn'+str(pars['fnoise'])
        
        
        
    
        
        
    
    #run
    for j, par in enumerate(par_vec):
        pars['use_predefined_data'] = True if j > 0 else False 
        print('experiment:', iexp, 'run:', j)
        pars[pars['par_label']] = par  
        comment = pars['par_label'] + str(par) + comment0
        folder0 = 'ds' + str(pars['x_opt']) + '_' + comment
        input_path = set_input_path(path0, folder0)  
        
        rand_init, s = init_random_seeds(s=42)
        exec(open('punc.py').read())

