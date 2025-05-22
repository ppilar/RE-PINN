# -*- coding: utf-8 -*-
import torch
import numpy as np


def RBF_kernel(x1, x2, h=1, opt='standard'):
    if type(h) is not torch.Tensor or h.ndim == 0:
        return 1/h*torch.exp(-1/(2*h**2)*torch.sum(((x1-x2)**2), -1))
    else:
        if opt=='factorized':
            return 1/h*torch.exp(-1/(2*h**2)*(x1-x2)**2)
        else:
            h = h**2
            xbuf = (x1-x2)@torch.diag(h**(-0.5))
            hnorm = torch.linalg.norm(h)
            return 1/hnorm**(-0.5) * torch.exp(-0.5*torch.sum(xbuf**2, -1))
    
def get_h(buf, opt='standard'):
    if type(buf) == np.ndarray:
        buf = torch.tensor(buf).float()
    if buf.ndim == 1:
        buf = buf.unsqueeze(1)
        
    if opt == 'factorized':
        diffs = torch.sqrt((buf.unsqueeze(1) - buf.unsqueeze(0))**2).permute(2,0,1) + 1e-7
        diffs = diffs.triu(diagonal=1).reshape(buf.shape[1],-1)
        diffs[diffs==0] = float('nan')
        h = diffs.nanmedian(1)[0]/np.sqrt(np.log(buf.shape[0]))
        return h
        
    else:
        diffs = torch.sqrt(torch.sum((buf.unsqueeze(1) - buf.unsqueeze(0))**2,2))
        triu_indices = diffs.triu(diagonal=1).nonzero().T
        triu_vec = diffs[triu_indices[0], triu_indices[1]]
        h = triu_vec.median().item()/np.sqrt(np.log(buf.shape[0]))
        return h
