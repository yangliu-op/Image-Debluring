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

def myMINRES(A, b, rtol, maxit, shift=0, reOrth=False, isZero=1E-6,
             print_warning=False):
    
    d, d = b.shape
    dv = b.device
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
    phi = beta1
    betan = beta1
    rk = b 
    vn = r3/betan
    Ar0 = AX(A, b, d).norm()
    
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
        # relres = rnorm / beta1
        Arnorml = phil * torch.sqrt(gbar**2 + dltan**2) # omit *\
        relAres = Arnorml/Ar0
        
        if relAres < rtol:
            flag = 5
        if iters >= maxit:
            flag = 4  ## exit before maxit
            if print_warning:
                print('Maximun iteration reached', flag, iters)
        if iters % 100 == 0 or flag != flag0:
            print(iters, relAres, Arnorml)
    return xl, rkl, relAres, iters, flag


def lsqr(A, b, rtol=1e-8, maxit=None, shift=0.0, conlim=1e8,
         show=False, calc_var=False, x0=None):
    
    d, d = b.shape
    b = b.reshape(d**2)

    # m, n = A.shape
    n = b.shape[0]
    if maxit is None:
        maxit = 2 * n
    var = torch.zeros_like(b)


    iters = 0
    flag = 0
    anorm = 0
    acond = 0
    shiftsq = shift**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    cs = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A*x,  alfa*v = A'*u.
    u = b
    bnorm = b.norm()
    x = torch.zeros_like(b)
    beta = bnorm

    if beta > 0:
        u = (1/beta) * u
        # v = A(u) # OC = 1
        v = AX(A, u, d)
        alfa = v.norm()
    else:
        v = x.clone()
        alfa = 0

    if alfa > 0:
        v = (1/alfa) * v
    w = v.clone()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        return x, flag, iters, r1norm, r2norm, anorm, acond, arnorm, xnorm, var



    # Main iteration loop.
    while iters < maxit:
        iters = iters + 1
        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alfa, v. These satisfy the relations
        #     beta*u  =  a*v   -  alfa*u,
        #     alfa*v  =  A'*u  -  beta*v.
        # u = A(v) - alfa * u # OC += 1
        u = AX(A, v, d) - alfa * u # OC += 1
        beta = u.norm()

        if beta > 0:
            u = (1/beta) * u
            anorm = torch.sqrt(anorm**2 + alfa**2 + beta**2 + shiftsq)
            # v = A(u) - beta * v # OC += 1
            v = AX(A,u,d) - beta * v # OC += 1
            alfa = v.norm()
            if alfa > 0:
                v = (1 / alfa) * v

        # Use a plane rotation to eliminate the shifting parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        if shift > 0:
            rhobar1 = torch.sqrt(rhobar**2 + shiftsq)
            cs1 = rhobar / rhobar1
            sn1 = shift / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            # cs1 = 1 and sn1 = 0
            rhobar1 = rhobar
            psi = 0.
        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = symGivens(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w
        
        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + dk.norm()**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate x.norm().
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = torch.sqrt(xxnorm + zbar**2)
        gamma = torch.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * torch.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = torch.sqrt(res1 + res2)
        arnorm = alfa * abs(tau)
        if iters == 1:
            Ar0 = arnorm
            relAres = 1
        else:
            relAres = arnorm/Ar0

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = torch.sqrt(r1norm^2 + shift^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = torch.sqrt(r2norm^2 - shift^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        if shift > 0:
            r1sq = rnorm**2 - shiftsq * xxnorm
            r1norm = torch.sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        if relAres <= rtol:
            flag = 1
        if iters % 100 == 0 or flag == 1:
            print(iters, relAres, arnorm)

        if flag != 0:
            break

    return x, b - AX(A,x,d), relAres, iters, flag

def MinresQLP(A, b, rtol=None, maxit=None, M=None, shift=None, reOrth=False, 
              isZero=1E-5, maxxnorm=None, Acondlim=None, TranCond=None, P=None):
    if rtol is None:
        rtol = 1e-4
    if maxit is None:
        maxit = 100
    if shift is None:
        shift = 0
    if maxxnorm is None:
        maxxnorm = 1e7
    if Acondlim is None:
        Acondlim = 1e15
    if TranCond is None:
        TranCond = 1e7
    d, d = b.shape
    n = d**2
    # device = b.device
    b = b.reshape(n)
    # rtol = rtol*shift/2
    x0 = torch.zeros_like(b)
    x = x0.clone()    
    Ab = AX(A, b, d)
    Ar0 = Ab.norm()
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)
    
    #function handle with M x r_hat = r
    if M is None:
        #test with M = lambda u: u
        noprecon = True
        pass
    else:
        noprecon = False
        r3 = Precond(M, r2)
        beta1 = torch.dot(r3, r2) #theta
        if beta1 <0:
            print('Error: "M" is indefinite!')
        else:
            beta1 = torch.sqrt(beta1)
    
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    QLPiter = 0
    beta = 0
    tau = 0
    taul = 0
    phi = beta1
    betan = beta1
    gmin = 0
    cs = -1
    sn = 0
    cr1 = -1
    sr1 = 0
    cr2 = -1
    sr2 = 0
    dltan = 0
    eplnn = 0
    gama = 0
    gamal = 0
    gamal2 = 0
    eta = 0
    etal = 0
    etal2 = 0
    vepln = 0
    veplnl = 0
    veplnl2 = 0
    ul3 = 0
    ul2 = 0
    ul = 0
    u = 0
    rnorm = betan
    xnorm = 0
    xl2norm = 0
    Anorm = 0
    Acond = 1
    relres = 1
    w = torch.zeros_like(b)
    wl = torch.zeros_like(b)
    rk = b #new
    
    if reOrth:
        Vn = b.reshape(-1, 1)/beta1
    #b = 0 --> x = 0 skip the main loop       
    while flag == flag0 and iters < maxit:
        #lanczos
        iters += 1
        betal = beta
        beta = betan
        v = r3/beta
        r3 = AX(A, v, d)
        if shift == 0:
            pass
        else:
            r3 = r3 - shift*v
        
        alfa = torch.dot(r3, v)
        
        if iters > 1:
            r3 = r3 - r1*beta/betal
            
        r3 = r3 - r2*alfa/beta
        if reOrth:
            r3 = r3 - torch.mv(Vn, torch.mv(Vn.T, r3))
        r1 = r2
        r2 = r3
        
        if noprecon:
            betan = torch.norm(r3)
            if iters == 1:
                if betan == 0:
                    if alfa == 0:
                        flag = 0
                        print('WARNNING: flag = 0')
                        break
                    else:
                        flag = -1
                        print('WARNNING: flag = -1')
                        # Probbaly lost all the info, x=0 is true solution
                        x = b/alfa
                        break
        else:
            r3 = Precond(M, r2)
            betan = torch.dot(r3, r2)
            if betan > 0:
                betan = torch.sqrt(betan)
            else:
                print('Error: "M" is indefinite or singular!')
                
        pnorm = torch.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)
        vnew = r3/betan #new
        if reOrth:
            Vn = torch.cat((Vn, vnew.reshape(-1, 1)), axis=1)
        
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        dlta_QLP = dlta
        #current left plane rotation Q_k
        gamal3 = gamal2
        gamal2 = gamal
        gamal = gama
        cs, sn, gama = symGivens(gbar, betan)
        gama_tmp = gama
        taul2 = taul
        taul = tau
        tau = cs*phi
        phil = phi
        phi = sn*phi
        #previous right plane rotation P_{k-2,k}
        if iters > 2:
            veplnl2 = veplnl
            etal2 = etal
            etal = eta
            dlta_tmp = sr2*vepln - cr2*dlta
            veplnl = cr2*vepln + sr2*dlta
            dlta = dlta_tmp
            eta = sr2*gama
            gama = -cr2 *gama
        #current right plane rotation P{k-1,k}
        if iters > 1:
            cr1, sr1, gamal = symGivens(gamal, dlta)
            vepln = sr1*gama
            gama = -cr1*gama
        
        ul4 = ul3
        ul3 = ul2
        if iters > 2:
            ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
        if iters > 1:
            ul = (taul - etal*ul3 - veplnl *ul2)/gamal
        xnorm_tmp = torch.sqrt(torch.tensor(xl2norm**2 + ul2**2 + ul**2))
        if abs(gama) > 1e-16 and xnorm_tmp < maxxnorm:
            u = (tau - eta*ul2 - vepln*ul)/gama
            if torch.sqrt(torch.tensor(xnorm_tmp**2 + u**2)) > maxxnorm:
                u = 0
                flag = 3
                print('WARNNING: flag = 3')
        else:
            u = 0
            flag = 6
        xl2norm = torch.sqrt(torch.tensor(xl2norm**2 + ul2**2))
        xnorm = torch.sqrt(torch.tensor(xl2norm**2 + ul**2 + u**2))
        #update w&x
        #Minres
        if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
            if xnorm < maxxnorm:
                x += tau*w
            else:
                flag = 3
                print('WARNNING: flag = 3')
        #Minres-QLP
        else:
            QLPiter += 1
            if QLPiter == 1:
                xl2 = torch.zeros_like(b)
                if (iters > 1):  # construct w_{k-3}, w_{k-2}, w_{k-1}
                    if iters > 3:
                        wl2 = gamal3*wl2 + veplnl2*wl + etal*w
                    if iters > 2:
                        wl = gamal_QLP*wl + vepln_QLP*w
                    w = gama_QLP*w
                    xl2 = x - wl*ul_QLP - w*u_QLP
                    
            if iters == 1:
                wl2 = wl
                wl = v*sr1
                w = -v*cr1                
            elif iters == 2:
                wl2 = wl
                wl = w*cr1 + v*sr1
                w = w*sr1 - v*cr1
            else:
                wl2 = wl
                wl = w
                w = wl2*sr2 - v*cr2
                wl2 = wl2*cr2 +v*sr2
                v = wl*cr1 + w*sr1
                w = wl*sr1 - w*cr1
                wl = v
            xl2 = xl2 + wl2*ul2
            xl = x
            x = xl2 + wl*ul + w*u 
            # print(P @ x - x)
            # new
            oldrk = rk
            rk = sn**2 * oldrk - phi * cs * vnew
            
        #next right plane rotation P{k-1,k+1}
        gamal_tmp = gamal
        cr2, sr2, gamal = symGivens(gamal, eplnn)
        #transfering from Minres to Minres-QLP
        gamal_QLP = gamal_tmp
        vepln_QLP = vepln
        gama_QLP = gama
        ul_QLP = ul
        u_QLP = u
        ## Estimate various norms
        abs_gama = abs(gama)
        Anorm = max([Anorm, pnorm, gamal, abs_gama])
        if iters == 1:
            gmin = gama
            gminl = gmin
        elif iters > 1:
            gminl2 = gminl
            gminl = gmin
            gmin = min([gminl2, gamal, abs_gama])
        Acondl = Acond
        Acond = Anorm / gmin
        rnorml = rnorm
        relresl = relres
        if flag != 9:
            rnorm = phi
        
        Arnorml = phil * torch.sqrt(gbar**2 + dltan**2)
        relAres = Arnorml/Ar0
        
        ## See if any of the stopping criteria are satisfied.
        if (flag == flag0) or (flag == 6):            
            if iters >= maxit:
                flag = 5 #exit before maxit
            if Acond >= Acondlim:
                flag = 4 #Huge Acond
                print('WARNNING: Acondlim exceeded!')
            if xnorm >= maxxnorm:
                flag = 3 #xnorm exceeded
                print('WARNNING: maxxnorm exceeded!')
            # if epsx >= beta1:
            #     flag = 2 #x = eigenvector
            if relAres <= rtol:
                flag = 1 #Trustful Solution
        if flag == 3 or flag == 4:
            print('WARNNING: possibly singular!')
            #possibly singular
            iters = iters - 1
            Acond = Acondl
            rnorm = rnorml
            relres = relresl  
            
    return xl, rk, relAres, iters, flag

def lsmr(A, b, rtol=1e-6, maxiter=None, damp=0.0, conlim=1e8, 
         show=False, x0=None):
    
    dim, dim = b.shape
    b = b.reshape(dim**2)


    # stores the num of singular values
#    minDim = min([m, n])

#    if maxiter is None:
#        maxiter = minDim

    u = b
    normb = b.norm()
    x = torch.zeros_like(b)
    beta = normb
    
    if beta > 0:
        u = (1 / beta) * u
#        v = A.rmatvec(u)
        v = AX(A, u, dim)
        alpha = v.norm()
    else:
        v = torch.zeros_like(b)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    iters = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.clone()
    hbar = torch.zeros_like(b)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = torch.sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    flag = 0
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        return x, flag, iters, normr, normar, normA, condA, normx

    # Main iteration loop.
    while iters < maxiter:
        iters = iters + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

#        u = A.matvec(v) - alpha * u
        u = AX(A, v, dim) - alpha * u
        beta = u.norm()

        if beta > 0:
            u = (1 / beta) * u
#            v = A.rmatvec(u) - beta * v
            v = AX(A,u,dim) - beta*v
            alpha = v.norm()
            if alpha > 0:
                v = (1 / alpha) * v

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = symGivens(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = symGivens(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = symGivens(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = symGivens(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = torch.sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = torch.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if iters > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        if iters == 1:
            Ar0 = normar
            relAres = 1
        else:
            relAres = normar/Ar0
        normx = x.norm()

        if relAres <= rtol:
            flag = 1
        if iters % 100 == 0 or flag == 1:
            print(iters, relAres, normar)

        if flag > 0:
            break
    return x, b - AX(A,x,dim), relAres, iters, flag

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
        T += 1
        Ap = AX(A, p, d)
        pAp = torch.dot(p, Ap)
        if pAp < 0:
            print('pAp =', pAp)
            flag = 2
            # break
            # raise ValueError('pAp < 0 in myCG')
        alpha = delta/pAp
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


def Precond(M, r):
    if callable(M):
        h = myCG(M, r, 10e-6, 1000)[0]
#        h = cg(M, r)
    else:
        h = torch.mv(M, r)
    return h


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