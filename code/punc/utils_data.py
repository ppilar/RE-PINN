# -*- coding: utf-8 -*-

import numpy as np
import torch
import scipy
import scipy.io
import torch.nn as nn
import csv

import matplotlib.pyplot as plt

def get_rwalk(Nsteps, rmax = 10, rmin = -11):
    x = 0
    rwalk = [x]
    for j in range(Nsteps-1):
        buf = rwalk[-1] + np.random.randint(-2,3,1)
        if buf > rmax or buf < rmin:
            buf = rwalk[-1]            
        rwalk.append(buf.item())
    return rwalk

def get_xtmode(imax, opt='fixed', Nrandom = 4):
    if opt == 'fixed':
        mt = int(2*imax/3)
        dt = int(imax/10)
        inds = [0, mt-dt, mt, mt+dt]
    elif opt == 'random':
        rng = np.random.default_rng()
        inds = list(rng.choice(imax, size=Nrandom, replace=False))
    return inds

def get_cluster(mean, std, N):
    t0 = mean[0]
    x0 = mean[1]
    ix = x0 + np.random.randint(-std[0],std[1]+1,N)
    it = t0 + np.random.randint(-std[0],std[1]+1,N)
    return ix, it

def get_random_clusters(Nt, Nx, Ncluster=3, N=5):
    ix = []
    it = []
    for j in range(Ncluster):
        mean = (np.random.randint(2,Nt-2,1), np.random.randint(2,Nx-2,1))
        std = (np.random.randint(1,int(Nt/6),1), np.random.randint(1,int(Nx/6),1))
        print(mean)
        print(std)
        ix_buf, it_buf = get_cluster(mean, std, N)
        ix += list(np.minimum(ix_buf, Nx-1))
        it += list(np.minimum(it_buf, Nt-1))
        
    return np.array(ix), np.array(it)

def get_slice(t0, x0, dx, dt, N):
    ix = x0 + np.linspace(0,N-1,N, dtype=int)*dx
    it = t0 + np.linspace(0,N-1,N, dtype=int)*dt
    return ix, it
    
def get_random_slices(Nt, Nx, Nslices, N):    
    ix = []
    it = []
    for j in range(Nslices):
        t0 = np.random.randint(0,Nt-2,1)
        x0 = np.random.randint(2,Nx-2,1)
        dx = np.random.choice([-2,-1,0,1,2])
        dt = np.random.choice([1,2])
        ix_buf, it_buf = get_slice(t0, x0, dx, dt, N)
        ix += list(ix_buf)
        it += list(it_buf)
        
    return np.array(ix), np.array(it)
    
def check_indices(ix, it, Nt, Nx):
    inds = np.concatenate((np.where(ix < 0)[0], np.where(ix > Nx-1)[0], np.where(it < 0)[0], np.where(it > Nt-1)[0]))
    
    ix = np.delete(ix, inds)
    it = np.delete(it, inds)
    return ix, it
    

def load_hare_lynx_data(pars, opt='numpy'):
    with open('../data/hudson-bay-lynx-hare.csv', 'r', newline='', encoding='utf-8') as file:
        data = []
        reader = csv.reader(file)
        for j, row in enumerate(reader):
            if j > 2:
                data.append([float(n) for n in row])
            print(row)
        data = np.array(data)
        
    plt.plot(data[:,0], data[:,1:])

    N_train = pars['N_train'] if pars['N_train'] < 21 else 15
    N_test = 21 - N_train
    ind_train = sorted(np.random.choice(data.shape[0], size=N_train, replace=False))
    ind_test = sorted(np.setdiff1d(np.arange(data.shape[0]), ind_train))

    t_train = data[ind_train,0]
    t_test = data[ind_test,0]
    y_train = data[ind_train,1:]
    y_test = data[ind_test,1:]
        
    if opt == 'numpy' or opt == 'np':
        return t_train, y_train, t_test, y_test
    else:
        return torch.tensor(t_train).float(), torch.tensor(y_train).float(), torch.tensor(t_test).float(), torch.tensor(y_test).float()
        
        

def load_advection_data(pars):
    if pars['x_opt'] == 102:
        fname = '1D_Advection_Sols_beta0.2'
    elif pars['x_opt'] == 103:
        fname = '1D_Burgers_Sols_Nu0.01'
        
        
    with open('../data/' + fname + '.npy', 'rb') as f:
        data = np.load(f)
        t = data['t']
        x = data['x']
        y = data['sol']
        
    buf = np.meshgrid(t.squeeze(),x.squeeze())
    tbuf = buf[0][::50, ::10].T
    xbuf = buf[1][::50, ::10].T
    ybuf = y[pars['isample']].T[::50, ::10].T
    
    Nt = tbuf.shape[0]
    Nx = tbuf.shape[1]
    
    if pars['measurement_pattern'] == 'tmode':
        #it = [0,10,12,14]
        it = get_xtmode(ybuf.shape[0], opt=pars['measurement_pattern_opt'])
        tbuf = tbuf[it,:]
        xbuf = xbuf[it,:]
        ybuf = ybuf[it,:]
    elif pars['measurement_pattern'] == 'xmode':
        ix = get_xtmode(ybuf.shape[1], opt=pars['measurement_pattern_opt'])
        tbuf = tbuf[:,ix]
        xbuf = xbuf[:,ix]
        ybuf = ybuf[:,ix]
    elif pars['measurement_pattern'] == 'xtmode':
        buf1 = np.random.randint(5)
        buf2 = 4 - buf1
        it = get_xtmode(ybuf.shape[0], opt=pars['measurement_pattern_opt'], Nrandom = buf1)
        ix = get_xtmode(ybuf.shape[1], opt=pars['measurement_pattern_opt'], Nrandom = buf2)
        
        tbuf = np.concatenate((tbuf[it,:].reshape(-1), tbuf[:,ix].reshape(-1)),0)
        xbuf = np.concatenate((xbuf[it,:].reshape(-1), xbuf[:,ix].reshape(-1)),0)
        ybuf = np.concatenate((ybuf[it,:].reshape(-1), ybuf[:,ix].reshape(-1)),0)
        
        
    elif pars['measurement_pattern'] == 'clusters':
        ix, it = get_random_clusters(Nt, Nx, 4, 5)
        ix, it = check_indices(ix, it, Nt, Nx)
        tbuf = tbuf[it,ix]
        xbuf = xbuf[it,ix]
        ybuf = ybuf[it,ix]
    elif pars['measurement_pattern'] == 'slices':
        ix, it = get_random_slices(Nt, Nx, 4, 10)
        ix, it = check_indices(ix, it, Nt, Nx)
              
        tbuf = tbuf[it,ix]
        xbuf = xbuf[it,ix]
        ybuf = ybuf[it,ix]
    elif pars['measurement_pattern'] == 'boundary':
        tlist = [0]*Nx + list(range(0, Nt)) + list(range(0, Nt))
        xlist = list(range(0, Nx)) + [0]*Nt + [-1]*Nt       
        
    elif pars['measurement_pattern'] == 'triangle':    
        xlist = list(range(0,20,2)) + [20] +  list(reversed(range(0,20,2)))
        tlist = list(range(0,21,1))
    elif pars['measurement_pattern'] == 'diagonal':
        xlist = list(range(0,21,1)) + list(range(3,21,1)) + list(range(0,18,1))
        tlist = list(range(0,21,1)) + list(range(0,18,1)) + list(range(3,21,1))
    elif pars['measurement_pattern'] == 'random':
        tlist = list(np.random.randint(0,tbuf.shape[0],pars['N_train']))        
        xlist = list(np.random.randint(0,tbuf.shape[1],pars['N_train']))
    elif pars['measurement_pattern'] == 'rwalk':
        x0vec = [11, 8, 14]
        tlist = list(range(0,21))*len(x0vec)
        xlist = []
        for j in range(len(x0vec)):
            xlist += list(x0vec[j]*np.ones(21, dtype=int) + get_rwalk(21))
      
    if not pars['measurement_pattern'] in ['tmode', 'xmode', 'xtmode', 'clusters', 'slices']:
        tbuf = tbuf[tlist, xlist]
        xbuf = xbuf[tlist, xlist]
        ybuf = ybuf[tlist, xlist]
    
    
    train_in = torch.cat((torch.tensor(tbuf).reshape(-1,1), torch.tensor(xbuf).reshape(-1,1)), 1)    
    train_out = (torch.tensor(ybuf)).reshape(-1)
    
    
    tbuf2 = buf[0][::30, ::6].T
    xbuf2 = buf[1][::30, ::6].T
    test_in = torch.cat((torch.tensor(tbuf2).reshape(-1,1), torch.tensor(xbuf2).reshape(-1,1)), 1)
    test_out = (torch.tensor(y[pars['isample']].T[::30, ::6]).T).reshape(-1)
    
    pars['tshape'] = list(tbuf2.shape)
        
    return train_in, train_out, test_in, test_out
    
    
def load_exp_data(pars):
    t_train, t_test, t_coll = get_ttrain_tcoll(pars['tmin'], pars['tmax'], pars['tmin_coll'], pars['tmax_coll'], pars['N_train'], pars['N_coll'])
    x_train = fx(t_train)
    f_train = fmeq0(x_train)    
    x_test = fx(t_test)
    f_test = fmeq0(x_test)
    return t_train, f_train, t_test, f_test