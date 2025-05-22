#define settings
if not 'pars' in locals(): pars = dict() #dictionary that contains training parameters and settings


# x_opt - choose DE: 
    #1 ... exp_1d,  
    #5 ... dHO_1d, 
    #7 ... Lotka-Volterra 
    #102 ... advection
    
# model_vec -choose models:
    #0 ... base, 
    #1 ... repulsive (only f), 
    #2 ... repulsive (only lambda), 
    #3 ... repulsive (f and l), 
    #4 ... repulsive (f and l, different h), 
    #5 ... repulsive (f and l, factorized, 
    #6 ... repulsive (f and l, fully factorized)


init_par(pars, 'x_opt', 5) #which DE to solve
init_par(pars, 'model_vec', [0,5,6]) #models to be trained
init_par(pars, 'Npinn', 500) #number of training iterations
init_par(pars, 'Nrun', 1) #number of runs to average over


init_par(pars, 'iplot', 1000) #after how many iterations to make plot of current predictions
init_par(pars, 'N_ensemble', 10)




def init_standard_pars(pars):
    if pars['x_opt'] == 1:
        init_par(pars, 'N_train', 10)
        init_par(pars, 'N_test', 20)
        init_par(pars, 'N_coll', 50)
        init_par(pars, 'ld_fac', 1)
        init_par(pars, 'lf_fac', 5)
        init_par(pars, 'fnoise', 2) 
        
    if pars['x_opt'] == 2:        
        init_par(pars, 'learn_rhs', 1)
        init_par(pars, 'N_train_rhs', 12)
        if pars['learn_rhs'] == 1:
            init_par(pars, 'N_train', 4)
        else:
            init_par(pars, 'N_train', 8)
        init_par(pars, 'N_test', 20)
        init_par(pars, 'N_coll', 50)
        
    if pars['x_opt'] == 5:       
        init_par(pars, 'N_train', 10)
        init_par(pars, 'N_test', 20)
        init_par(pars, 'N_coll', 100)
        init_par(pars, 'ld_fac', 1)
        init_par(pars, 'lf_fac', 1)
        init_par(pars, 'fnoise', 1) 
        init_par(pars, 'priors', ['l'])        
        init_par(pars, 'i_start_repulsion', 3000)
        init_par(pars, 't_opt', 'gap')
        
    if pars['x_opt'] == 7:
        init_par(pars, 'fnoise', 1.)
        init_par(pars, 'lf_fac', 0.5)
        init_par(pars, 'N_train', 12)
        init_par(pars, 'priors', ['l'])
        init_par(pars, 'fnoise', 1)
        init_par(pars, 'real_data', 0)
        init_par(pars, 't_opt', 'gap')
        init_par(pars, 'i_start_repulsion', 3000)
        
        
    if pars['x_opt'] == 102:
        init_par(pars, 'N_train', 64)
        init_par(pars, 'N_test', 64)
        init_par(pars, 'N_coll', 256)
        init_par(pars, 'ld_fac', 1)
        init_par(pars, 'lf_fac', 0.1)    
        init_par(pars, 'collocation_base', 'Sobol')        
        init_par(pars, 'lr_pinn', 0.003)
        init_par(pars, 'weight_decay', 5e-4)
        init_par(pars, 'priors', ['l'])
        init_par(pars, 'measurement_pattern', 'random')
        init_par(pars, 'measurement_pattern_opt', 'random')
        
    
    init_par(pars, 'learn_rhs', 0)
    init_par(pars, 'N_test_rhs', 20)
    init_par(pars, 'ld_fac', 1)
    init_par(pars, 'lf_fac', 1)
    init_par(pars, 'fnoise', 1)   
    init_par(pars, 'fnoise_rhs', 1)
    init_par(pars, 'bs_train', 64)
    init_par(pars, 'bs_coll', 256)    
    init_par(pars, 'repulsive_approximation', 'KDE')
    init_par(pars, 'sigma_w', 10.)
    init_par(pars, 'priors', [])
    init_par(pars, 'repulsion_space', 'f')
    init_par(pars, 'lf_schedule', 1)
    init_par(pars, 'i_start_repulsion', 0)
    init_par(pars, 'real_data', 0)
    init_par(pars, 't_opt', 'linspace')
    
    
    #traning and eval
    buf = int(1*pars['Npinn']) if pars['x_opt'] != 101 else int(0.8*pars['Npinn'])  #after how many iterations to take scheduler step
    init_par(pars, 'i_sched', buf)
    buf = 100 if pars['x_opt'] != 101 else 500 #after how many iterations to (repeatedly) calculate statistics on test data
    init_par(pars, 'itest', buf)
    
    
    
    #misc
    init_par(pars, 'collocation_opt', 'base')
    init_par(pars, 'collocation_base', 'grid')
    init_par(pars, 'adaptive_weights', 0)
    init_par(pars, 'n_opt', 'G')  #choose noise type; 'G'
    init_par(pars, 'Nmodel', 9) #number of different models (that could be part of the current selection)
    init_par(pars, 'par_label', 'none')
    init_par(pars, 'activation', 'tanh')
    init_par(pars, 'weight_decay', 0)
    init_par(pars, 'load_data', False)
    init_par(pars, 'learn_pde_pars', 1) #1 ... learn pde pars; 0 ... use correct pars
    init_par(pars, 'lr_pinn', 0.01)
    init_par(pars, 'f_prior', 'uniform') #'standard', 'uniform'
    init_par(pars, 'w_prior', 'standard')
    init_par(pars, 'isample', 4)
    if pars['learn_rhs'] == 0:
        pars['N_train_rhs'] = 0 
        pars['N_test_rhs'] = 0
    init_par(pars, 'add_repulsion', 0)
        
        
    

## init values
init_standard_pars(pars)


#data loading
init_par(pars, 'use_predefined_data', False)
init_par(pars, 'temp_folder', 'experiments1')

if not 'comment' in locals(): comment = '_fn'+str(pars['fnoise'])