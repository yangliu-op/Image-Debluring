# -*- coding: utf-8 -*-
"""
Created on Sat May  7 08:31:21 2022

@author: Yang
"""
import torch


def AX(A, x, d):
    X = x.reshape(d, d)
    Ax = A(X)
    return Ax.reshape(d**2)

def myMINRES(A, b, rtol, maxit, shift=0, reOrth=False, isZero=1E-5,
             print_warning=False):
    
    d, d = b.shape
    dv = b.device
    maxnorm = 1/isZero
    b = b.reshape(d**2)
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
        
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    beta = 0
    tau = 0
    cs = -1
    sn = 0
    dltan = 0
    eplnn = 0
    gama = 0
    xnorm = 0
    phi = beta1
    relres = phi / beta1
    betan = beta1
    rnorm = betan
    rk = b 
    vn = r3/betan
    
    x = torch.zeros_like(b)
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b) 
    
    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 9        
        
    if reOrth:
        Vn = vn.reshape(-1, 1)
        
    while flag == flag0:
        #lanczos
        betal = beta
        beta = betan
        v = vn
        r3 = AX(A, v, d)
        iters += 1
        
        if shift == 0:
            pass
        else:
            r3 = r3 + shift*v
        
        alfa = torch.dot(r3, v)
        if iters > 1:
            r3 = r3 - r1*beta/betal            
        r3 = r3 - r2*alfa/beta
        
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
        
        betan = torch.norm(r3)
        if iters == 1:
            if betan < isZero:
                if alfa == 0:
                    flag = 0
                    if print_warning:
                        print('flag = 0')
                    break
                else:
                    flag = -1
                    x = b/alfa
                    if print_warning:
                        print('flag = -1')
                    break
                
            
        vn = r3/betan
        if reOrth:
            Vn = torch.cat((Vn, vn.reshape(-1, 1)), axis=1)
                
        ## QR decomposition
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        
        csl = cs
        phil = phi
        rkl = rk
        xl = x  
            
        ## Check if Lanczos Tridiagonal matrix T is singular
        cs, sn, gama = symGivens(gbar, betan, device=dv) 
        if gama > isZero:
            tau = cs*phi
            phi = sn*phi
            
            ## update w&x
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta*wl)/gama
            x = x + tau*w
            rk = sn**2 * rkl - phi * cs * vn
        else:
            ## if gbar = betan = 0, Lanczos Tridiagonal matrix T is singular
            ## system inconsitent, b is not in the range of A,
            ## MINRES terminate with phi \neq 0.
            cs = 0
            sn = 1
            gama = 0  
            phi = phil
            rk = rkl
            x = xl
            flag = 2
            if print_warning:
                print('flag = 2, b is not in the range(A)!')
            maxit += 1
            
        ## stopping criteria
        # xnorm = torch.norm(x)   
        # print(iters, xnorm)
        rnorm = phi
        relres = rnorm / beta1
        # Arnorml = gbar*v + dltan*vn # omit *phil
        # relres_Ar = Arnorml.norm()/Anorm
        # maxx = torch.max(abs(x))
        # if gama < isZero or betan < isZero or phi/x.norm() < rtol:
        #     flag = 5
        if iters >= maxit:
            flag = 4  ## exit before maxit
            if print_warning:
                print('Maximun iteration reached', flag, iters)
        if iters % 100 == 0 or flag != flag0:
            print(iters, x.norm(), betan, gama, relres, phi/x.norm(), flag)
    return x, rk, relres, iters, flag

def myCG(A, b, tol, maxiter, reOrth=False, isZero=1E-15):
#def myCG(A, b, tol, maxiter, xx=None, L=None):
    """
    Conjugate gradient mathods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
    OUTPUT:
        x: best solution x
        rel_res: relative residual
        T: iterations
    """
    d, d = b.shape
    # dv = b.device
    b = b.reshape(d**2)
    x = torch.zeros_like(b)
    r = b
    T = 0
    rel_res_best = 1
    rel_res = 1
    flag = 0
    delta = torch.dot(r, r)
    p = r.clone()
    
    while T < maxiter and rel_res >= tol:
        # print(T)
        T += 1
        Ap = AX(A, p, d)
        pAp = torch.dot(p, Ap)
        if pAp < 0:
            print('pAp =', pAp)
            flag = 2
            # break
            # raise ValueError('pAp < 0 in myCG')
        alpha = delta/pAp
        xl = x
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = torch.norm(r)/torch.norm(b)            
        if rel_res_best > rel_res:
            rel_res_best = rel_res
        prev_delta = delta
        delta = torch.dot(r, r)
        p = r + (delta/prev_delta)*p
    return x, r, rel_res, T, flag

def house(A, d):
    m, n = A.shape
    R = torch.zeros(m, n,device=A.device, dtype=A.dtype)
    Q = torch.eye(m,device=A.device, dtype=A.dtype)
    R = torch.clone(A)
    it = min(min(m,n),d)
    for i in range(it):
        a1 = R[i:m,i]
        e1 = torch.zeros(m-i,device=A.device, dtype=A.dtype)
        e1[0] = 1
        u = a1 + torch.sign(a1[0])*a1.norm()*e1
        u = u/u.norm()
        R[i:m, i:n] = R[i:m, i:n] - torch.ger(2*u, torch.mv(R[i:m, i:n].T, u))
        P = torch.eye(m, dtype=A.dtype)
        P[i:m, i:m] = P[i:m, i:m] - torch.ger(2*u, u)
        Q = torch.mm(P, Q)
    return Q[:it,:].T, R[:it,:]

def symGivens(a, b, device="cpu"):
    """This is used by the function minresQLP.
    
    Arguments:
        a (float): A number.
        b (float): A number.
        device (torch.device): The device tensors will be allocated to.
    """
    if not torch.is_tensor(a):
        a = torch.tensor(float(a), device=device)
    if not torch.is_tensor(b):
        b = torch.tensor(float(b), device=device)
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = torch.sign(a)
        s = 0
        r = torch.abs(a)
    elif a == 0:
        c = 0
        s = torch.sign(b)
        r = torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
        t = a / b
        s = torch.sign(b) / torch.sqrt(1 + t ** 2)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = torch.sign(a) / torch.sqrt(1 + t ** 2)
        s = c * t
        r = a / c
    return c, s, r