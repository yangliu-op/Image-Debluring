# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:14:08 2022

@author: Yang

Image debluring experiment for paper:
Obtaining Pseudo-inverse Solutions With MINRES
https://arxiv.org/abs/2309.17096
Authors: Yang Liu, Andre Milzarek, Fred Roosta 
"""

import torch
from PIL import Image
import numpy as np
import math
import os
from scipy.linalg import toeplitz
from scipy.sparse import kron
from lsqr import lsqr
from solvers import house
import matplotlib.pyplot as plt
from show_deblurring import show_deblurring, normalize
from time import time

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Pblurring(object):
    # Construct Gaussian bluring matrix class
    def __init__(self, n, bandw, sigma, dtype=torch.float64, device="cpu"):
        self.dv = device
        self.dt = dtype
        bandw = min(bandw,n)
        tmp = torch.exp(-torch.arange(0,bandw)**2/sigma/sigma/2)
        z = np.zeros(n)
        z[:bandw] = tmp
        npT1 = toeplitz(z)/np.sqrt(2*math.pi)/sigma
        T1 = torch.tensor(npT1, dtype=self.dt, device=self.dv)
        self.Toeplitz = T1
        self.rank = n 
        t0 = time()
        self.U, self.s, self.V = torch.svd(T1) # for truncated svd methods
        self.time = time() - t0
    
    def range_(self, index):
        img = lambda X: self.U[:,index] @ X @ self.U[:,index].T
        return img
    
    def kron_(self):
        # Function handle of applying Gaussian bluring matrix to an image X
        T = self.Toeplitz
        img = lambda X: T @ X @ T
        return img
    
    def A_(self): 
        # represent Gaussian debluring matrix as a Kronecker produck of
        # Two Toeplitz matrix
        return kron(self.Toeplitz, self.Toeplitz).todense()
    
    def kron_G_vec(self, C, re=False): 
        # Function handle of using preconditioner C
        if re:
            img = lambda tX: C @ tX @ C.T # S @ tx
        else:
            img = lambda B: C.T @ B @ C # S.T @ b
        return img
    
    def kron_AG_vec(self, C, re=False):
        # Function handle of using range-preseved preconditioner C
        T = C.T @ self.Toeplitz
        if re:
            img = lambda tX: T.T @ tX @ T # S @ tx
        else:
            img = lambda B: T @ B @ T.T # S.T @ b
        return img
    
    def kron_G(self, C):
        # Explicit methods using preconditioner C
        T = self.Toeplitz
        T2 = C.T @ (T @ C)
        img = lambda X: T2 @ X @ T2
        return img
    
    def kron_AG(self, C):
        # Explicit methods using range-preseved preconditioner C
        T = self.Toeplitz
        T2 = C.T @ (T @ (T @ (T @ C)))
        img = lambda X: T2 @ X @ T2
        return img

def plifting(x, r, br):
    # Lifting formula of precontioned MINRES
    return (x - (r@x)/(br@r) * br)

def lifting(x, r):
    # Lifting formula of MINRES
    return (x - (r@x)/(r@r) * r)

# import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.measure import compare_mse, compare_psnr, compare_ssmi

def myMeasure(img, img0):
    # Display PSNR score and SSIM score
    return 'PSNR (%.4s)  SSIM (%s%%)' % (psnr(img, img0).item(), 
                                     int(100*ssim(img, img0, multichannel=True).item()))

def TSVD(U, s, V, B_r, B_g, B_b, epsilon=0):
    # Truncated SVD
    sd = torch.zeros_like(s)
    index = s>epsilon # smallest singlar values cutoff
    s_tmp = s[index]
    sd[index] = 1/s_tmp
    Td = V @ torch.diag(sd) @ U.T
    d, d = B_r.shape
    img = np.zeros((d, d, 3))
    img[:,:,0] = normalize(Td @ B_r @ Td.T) # red channel
    img[:,:,1] = normalize(Td @ B_g @ Td.T) # green channel
    img[:,:,2] = normalize(Td @ B_b @ Td.T) # blue channel
    return img
    
def main():
    mypath = 'blur_'
        
    dt = torch.float64
    # dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dv = torch.device("cpu")
    maxit = 30
    rtol = 1E-16
    re = False # Reorthogonalization
    ze = 1E-6 # Treat as zero cutoff
    noise_scale = 1 # Scaling randn noise
    
    LSQR = True # LSQR benchmark
    MR = True # MINRES and Lifted MINRES methods
    PMR = True # Preconditioned MINRES methods
    SVD = True # Truncated SVD benchmark
    
    ### choose image: your-awesome-image
    pic = '20220607225846.jpg'
    img = Image.open('test_images/' + pic)
    # img = img.transpose(Image.ROTATE_90)
    
    filename = mypath  + pic
    if not os.path.isdir(filename):
        os.makedirs(filename)
    
    ### Get RGB matrices of image
    data = np.asarray(img)
    red = torch.tensor(data[:,:,0], dtype=dt, device=dv)
    green = torch.tensor(data[:,:,1], dtype=dt, device=dv)
    blue = torch.tensor(data[:,:,2], dtype=dt, device=dv)
    # imgrgb = img.covert('RGB')
    # red = np.array(imgrgb.getchannel(0))
    # green = np.array(imgrgb.getchannel(1))
    # blue = np.array(imgrgb.getchannel(2))
    
    ### Get gray matrices of image
    # imggray = img.convert('L')
    # imgmat = np.array(list(imggray.getdata()), float)
    # imgmat.shape = (imggray.size[1], imggray.size[0])
    # imgmat = np.matrix(imgmat)
    
    # Figure setting
    figsz1 = (20,4) # single row figure size
    figsz2 = (20,8.1) # double row figure size
    mydpi = 100
    fsize = 16
    fig1, axs1 = plt.subplots(1, 5, constrained_layout=False, sharey=True, 
                              figsize=figsz1)    
    # plt.imshow(imggray, cmap='gray')
    # plt.imshow(data)
    # plt.title('Image')
    original_img = data/255
    
    ax = axs1[0]
    ax.imshow(data)
    ax.set_ylabel('Image Deblurring', fontsize=fsize)
    ax.set_xticklabels([])
    ax.set_xlabel('Original Image', fontsize=fsize)
    # ax.set_title(myMeasure(original_img, original_img))
    
    ### select pixels/region option
    # n1 = 302
    # d = 400
    # n2 = n1 + d
    # red = red[n1:n2,n1-1:n2-1]
    # green = green[n1:n2,n1-1:n2-1]
    # blue = blue[n1:n2,n1-1:n2-1]
    # original_img = data[n1:n2,n1-1:n2-1,:]/255
    
# =============================================================================
#     Initialize 
# =============================================================================
    d, d = red.shape
    # define Gaussian bluring operator (full rank but extremely ill-conditioned)
    myA = Pblurring(d, bandw=101, sigma=9, dtype=dt, device=dv) 
    # Gassian bluring operator as a function handle on image channle using 
    # Kronecker products
    A = myA.kron_() 
    A_r = A(red)
    A_g = A(green)
    A_b = A(blue)
    
    # Add noise
    noise_r = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    noise_g = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    noise_b = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    B_r = A_r + noise_r
    B_g = A_g + noise_g
    B_b = A_b + noise_b
    
    # display blurred noised images using [0,1]-mapping
    blurred_noised_img = np.zeros((d,d,3))
    blurred_noised_img[:,:,0] = normalize(B_r)
    blurred_noised_img[:,:,1] = normalize(B_g)
    blurred_noised_img[:,:,2] = normalize(B_b)
    
    ax = axs1[1]
    ax.imshow(blurred_noised_img)
    ax.set_xticklabels([])
    ax.set_xlabel('Noisy Blurred Image', fontsize=fsize)
    
    if LSQR: # using LSQR methods
        t1 = time()
        maxit_lsqr = maxit
        lsqr_r = lsqr(A, B_r, iter_lim=maxit_lsqr)[0]
        lsqr_g = lsqr(A, B_g, iter_lim=maxit_lsqr)[0]
        lsqr_b = lsqr(A, B_b, iter_lim=maxit_lsqr)[0]
        deblurred_lsqr_X = np.zeros((d,d,3))
        deblurred_lsqr_X[:,:,0] = normalize(torch.tensor(lsqr_r).reshape(d,d))
        deblurred_lsqr_X[:,:,1] = normalize(torch.tensor(lsqr_g).reshape(d,d))
        deblurred_lsqr_X[:,:,2] = normalize(torch.tensor(lsqr_b).reshape(d,d))
        tt = 'LSQR ' + '(%.1f sec)' % (time() - t1)
        
        ax = axs1[2]
        ax.imshow(deblurred_lsqr_X)
        ax.set_xticklabels([])
        ax.set_xlabel(tt, fontsize=fsize)
        ax.set_title(myMeasure(deblurred_lsqr_X, original_img), fontsize=fsize)
        print('lsqr', time() - t1)
    
    if MR: # MINRES & lifted MINRES methods
        img, img_lifted, tim = show_deblurring(A, B_r, tb_g=B_g, tb_b =B_b, 
                                               maxit=maxit, rtol=rtol,
                                               re=re, ze=ze)
        tt = 'MINRES ' + '(%.1f sec)' % (tim)
        
        ax = axs1[3]
        ax.imshow(img)
        ax.set_xticklabels([])
        ax.set_xlabel(tt, fontsize=fsize)
        ax.set_title(myMeasure(img, original_img), fontsize=fsize)
        
        tt = 'MINRES (Lifted)'
        plt.title(tt, fontsize=fsize)      
        plt.imshow(img_lifted)
        
        ax = axs1[4]
        ax.imshow(img_lifted)
        ax.set_xticklabels([])
        ax.set_xlabel(tt, fontsize=fsize)
        ax.set_title(myMeasure(img_lifted, original_img), fontsize=fsize)
        
    fig1.savefig(os.path.join(filename, 'deblurring'), dpi=mydpi)
  
        
    #%%    
    r = myA.rank
    d_tmp1 = [0.005, 0.01, 0.02, 0.1, 1] # Rank percentage with random preconditioner
    d_tmp2 = [0.2, 0.4, 0.6, 0.8, 1] # Rank percentage with range preserved preconditioner
    
    if PMR:
        W = torch.randn(d, d, dtype=dt, device=dv)           
        fig3, axs3 = plt.subplots(2, 5, constrained_layout=False, sharey=True, figsize=figsz2)
        for i in range(5):
            r_tmp = int(np.floor(r * np.sqrt(d_tmp2[i])))
            tim1 = time()
            Q2, R = house(W, r_tmp) # incomplete QR decomposition on W
            C = Q2 @ np.diag(np.linspace(2, 1, r_tmp)) # set condition number
            tA = myA.kron_G(C) # PMR subproblem underlying matrix tA = S.T A S
            tb = myA.kron_G_vec(C) # PMR subproblem RHS tb = S.T b
            S = myA.kron_G_vec(C, re=True) # PMR subproblem S
            tim2 = time() - tim1
            tim3 = time()
            img, img_lifted, tim = show_deblurring(tA, tb(B_r), S=S, tb_g=tb(B_g), 
                                                   tb_b =tb(B_b), maxit=maxit, 
                                                   rtol=rtol, re=re, ze=ze)
            tim4 = time() - tim3
            ax = axs3[0,i]
            ax.imshow(img)
            if i == 0:
                ax.set_ylabel(r'$ Range(S_2) \nsubseteq Range(A) $', fontsize=fsize)
            ax.set_xticklabels([])
            ax.set_title(myMeasure(img, original_img), fontsize=fsize)
            
            ax = axs3[1,i]
            ax.imshow(img_lifted)
            if i == 0:
                ax.set_ylabel('Lifted of Above', fontsize=fsize)
            tt = 'Rank %s%% (%.1f/%.1f sec)' % (int(d_tmp2[i]*100), tim4, tim2)
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img_lifted, original_img), fontsize=fsize)
        
        fig3.savefig(os.path.join(filename, 'PMR_deblurring'), dpi=mydpi)
        #%% Range preserved PMR deblurring
        fig2, axs2 = plt.subplots(2, 5, constrained_layout=False, sharey=True, figsize=figsz2)
        for i in range(5):
            r_tmp = int(np.floor(r * np.sqrt(d_tmp1[i])))
            tim1 = time()
            Q1, R = house(myA.Toeplitz @ W, r_tmp) # incomplete QR decomposition on AW
            C = Q1 @ np.diag(np.linspace(2, 1, r_tmp)) # set condition number
            tA = myA.kron_G(C) # PMR subproblem underlying matrix tA = S.T A S
            tb = myA.kron_G_vec(C) # PMR subproblem RHS tb = S.T b
            S = myA.kron_G_vec(C, re=True) # PMR subproblem S
            tim2 = time() - tim1
            tim3 = time()
            img, img_lifted, tim = show_deblurring(tA, tb(B_r), S=S, tb_g=tb(B_g),
                                                   tb_b =tb(B_b), maxit=maxit, 
                                                   rtol=rtol, re=re, ze=ze)
            tim4 = time() - tim3
            ax = axs2[0,i]
            ax.imshow(img)
            if i == 0:
                ax.set_ylabel(r'$ Range(S_1) \subseteq Range(A) $', fontsize=fsize)
            ax.set_xticklabels([])
            ax.set_title(myMeasure(img, original_img), fontsize=fsize)
            
            ax = axs2[1,i]
            ax.imshow(img_lifted)
            if i == 0:
                ax.set_ylabel('Lifted of Above', fontsize=fsize)
            if i < 1:
                tt = 'Rank %.1f%% (%.1f/%.1f sec)' % (d_tmp1[i]*100, tim4, tim2)
            else:
                tt = 'Rank %s%% (%.1f/%.1f sec)' % (int(d_tmp1[i]*100), tim4, tim2)
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img_lifted, original_img), fontsize=fsize)
        
        fig2.savefig(os.path.join(filename, 'PMR_range_deblurring'), dpi=mydpi)
    #%% Trucated SVD benchmark
    r = myA.rank
    d_tmp1 = [0.005, 0.01, 0.012, 0.015, 0.02] # # Rank percentage of Truncated SVD
    if SVD:
        fig4, axs4 = plt.subplots(1, 5, constrained_layout=False, sharey=True, figsize=figsz1)
        for i in range(5):        
            r_tmp = int(np.floor(r * np.sqrt(d_tmp1[i])))
            t0 = time()
            img = TSVD(myA.U[:,:r_tmp], myA.s[:r_tmp],
                       myA.V[:,:r_tmp], B_r, B_g, B_b)     
            t3 = time() - t0 + myA.time
            ax = axs4[i]
            print('SVDtime', t3)
            ax.imshow(img)
            if i == 0:
                ax.set_ylabel('Truncated SVD', fontsize=fsize)
            if i == 0 or i == 2 or i == 3:
                tt = 'Rank %.1f%%' % (d_tmp1[i]*100)
            else:
                tt = 'Rank %s%%' % (int(d_tmp1[i]*100))
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img, original_img), fontsize=fsize)
            
        fig4.savefig(os.path.join(filename, 'Truncated_deblurring'), dpi=mydpi)
    
if __name__ == '__main__':
    main()
    