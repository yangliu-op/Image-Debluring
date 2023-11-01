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

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    # return X
    X = X - torch.min(X)
    return X/torch.max(X)
    

def show_deblurring(tA, tb, S=None, tb_g=None, tb_b=None, maxit=500, rtol=1E-14, 
                    re=False, ze=1E-6, solver=myMINRES):
    '''
    

    Parameters
    ----------
    tA : Symmetric Matrix d^2 x d^2, S.T A S, or function handle
        Underlying matrix of preconditioned subproblem
    tb : Squared image d x d, S.T b
        RHS of preconditioned subproblem (red channel for color image)
    S : rectangular subpreconditioner s.t. M = S S.T. The default is None.
    tb_g : Green channel for color image. The default is None.
    tb_b : Blue channel for color image. The default is None.
    maxit : Positive Integer, optional
        Maximum iteration. The default is 500.
    rtol : Double, optional
        termination tolerance. The default is 1E-14.
    re : Binary, optional
        Reorthogonalization. The default is False.
    ze : Double, optional
        Treated as zero cutoff. The default is 1E-6.
    solver : function, optional
        Solver for solving the symmtric least-squares problem. 
        The default is myMINRES.

    Returns
    -------
    img : d x d x 3 color image using MINRES
    img_lifted : d x d x 3 color image using MINRES
        ref: Obtaining Pseudo-inverse Solutions With MINRES
        https://arxiv.org/abs/2309.17096
        Authors: Yang Liu, Andre Milzarek, Fred Roosta 
    time

    '''
    td, td = tb.shape
    t1 = time()
    tx, r, _, iters, _  = solver(tA, tb, rtol, maxit, reOrth=re, isZero=ze)
    pic_x = tx.reshape(td,td)
    tx_lifted = lifting(tx, r)
    pic_x_lifted = tx_lifted.reshape(td,td)
    if S is not None:
        pic_x = S(pic_x)
        pic_x_lifted = S(pic_x_lifted)
    d, d = pic_x.shape
    # print()
    
    if tb_g is not None:
        tx_g, r_g, _, iters_g, _  = solver(tA, tb_g, rtol, maxit, reOrth=re, isZero=ze)
        tx_b, r_b, _, iters, _  = solver(tA, tb_b, rtol, maxit, reOrth=re, isZero=ze)
        print(r.norm(), r_g.norm(), r_b.norm())
        pic_x_g = tx_g.reshape(td,td)
        pic_x_b = tx_b.reshape(td,td)
        if S is not None:
            pic_x_g = S(pic_x_g)
            pic_x_b = S(pic_x_b)     
        img = np.zeros((d, d, 3))
        img[:,:,0] = normalize(pic_x)
        img[:,:,1] = normalize(pic_x_g)
        img[:,:,2] = normalize(pic_x_b)
        tx_lifted_g = lifting(tx_g, r_g)
        tx_lifted_b = lifting(tx_b, r_b)
        pic_x_lifted_g = tx_lifted_g.reshape(td,td)
        pic_x_lifted_b = tx_lifted_b.reshape(td,td)
        if S is not None:
            pic_x_lifted_g = S(pic_x_lifted_g)
            pic_x_lifted_b = S(pic_x_lifted_b)
        img_lifted = np.zeros((d, d, 3))
        img_lifted[:,:,0] = normalize(pic_x_lifted)
        img_lifted[:,:,1] = normalize(pic_x_lifted_g)
        img_lifted[:,:,2] = normalize(pic_x_lifted_b)
        return img, img_lifted, time() - t1        
    else:
        img = normalize(pic_x)    
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        img_lifted = normalize(pic_x_lifted)
        return img, img_lifted, time() - t1