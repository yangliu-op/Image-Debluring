# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:14:57 2022

@author: Yang
"""
from solvers import myMINRES
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

def lifting(x, r):
    '''    

    Parameters
    ----------
    x : (exact) solution from MINRES
    r : residual from MINRES, b - A x

    Returns
    -------
    pseudo-inverse solution A^dagger b (
        or projection of x onto the range(A) if x is not exact solution)

    '''
    return (x - (x@r)/(r@r) * r)

def normalize(X):
    '''
    X --> [0,1]
    '''
    X = X - torch.min(X)
    return X/torch.max(X)
    

def deblurring_solve(tA, tb, Sd=None, tb_g=None, tb_b=None, maxit=500, rtol=1E-14, 
                    solver=myMINRES):
    td, td = tb.shape
    t1 = time()
    tx, r, _, iters, _  = solver(tA, tb, rtol, maxit)
    pic_x = tx.reshape(td,td)
    tx_lifted = lifting(tx, r)
    pic_x_lifted = tx_lifted.reshape(td,td)
    if Sd is not None:
        pic_x = Sd(pic_x)
        pic_x_lifted = Sd(pic_x_lifted)
    d, d = pic_x.shape
    average_iters = iters
    
    if tb_g is not None:
        tx_g, r_g, _, iters_g, _  = solver(tA, tb_g, rtol, maxit)
        tx_b, r_b, _, iters_b, _  = solver(tA, tb_b, rtol, maxit)
        pic_x_g = tx_g.reshape(td,td)
        pic_x_b = tx_b.reshape(td,td)
        if Sd is not None:
            pic_x_g = Sd(pic_x_g)
            pic_x_b = Sd(pic_x_b)     
        img = np.zeros((d, d, 3))
        img[:,:,0] = normalize(pic_x)
        img[:,:,1] = normalize(pic_x_g)
        img[:,:,2] = normalize(pic_x_b)
        tx_lifted_g = lifting(tx_g, r_g)
        tx_lifted_b = lifting(tx_b, r_b)
        pic_x_lifted_g = tx_lifted_g.reshape(td,td)
        pic_x_lifted_b = tx_lifted_b.reshape(td,td)
        if Sd is not None:
            pic_x_lifted_g = Sd(pic_x_lifted_g)
            pic_x_lifted_b = Sd(pic_x_lifted_b)
        img_lifted = np.zeros((d, d, 3))
        img_lifted[:,:,0] = normalize(pic_x_lifted)
        img_lifted[:,:,1] = normalize(pic_x_lifted_g)
        img_lifted[:,:,2] = normalize(pic_x_lifted_b)
        average_iters = (iters + iters_g + iters_b)/3 
    else:
        img = normalize(pic_x)    
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        img_lifted = normalize(pic_x_lifted)
    return img, img_lifted, time() - t1, average_iters 
    