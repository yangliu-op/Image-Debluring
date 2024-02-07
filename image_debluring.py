# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:18:32 2022

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
from solvers import myMINRES, house, lsqr, lsmr, MinresQLP
import matplotlib.pyplot as plt
from deblurring_solve import deblurring_solve, normalize
from time import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    ### Initialize
    mypath = 'blur_'
        
    dt = torch.float64
    dv = torch.device("cpu")
    maxit = 2
    rtol = 1E-5
    noise_scale = 1
    
    LSQR = True
    LSMR = True
    MINRESQLP = True
    MR = True
    PMR = True
    SVD = True
    
    ### use sysmetric images
    pic = '20220607225846.jpg'
    img_data= Image.open('test_images/' + pic) # use color images
    
    filename = mypath + pic
    if not os.path.isdir(filename):
        os.makedirs(filename)
    
    ### Get RGB matrices of image
    data = np.asarray(img_data)
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
    
    
    figsz1 = (16,4)
    figsz2 = (16,8.2)
    mydpi = 100
    fsize = 16
    fig0, axs0 = plt.subplots(1, 4, constrained_layout=False, sharey=True, figsize=figsz1)
    
    # plt.imshow(imggray, cmap='gray')
    # plt.imshow(data)
    # plt.title('Image')
    original_img = data/255
    
    ax = axs0[0]
    ax.imshow(original_img)
    # ax.set_ylabel('Image Deblurring', fontsize=fsize)
    ax.set_xticklabels([])
    ax.set_xlabel('Original Image', fontsize=fsize)
    # ax.set_title(myMeasure(original_img, original_img))
    
    ### select pixels if needed
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
    myA = Pblurring(d, bandw=100, sigma=10, dtype=dt, device=dv)
    A = myA.kron_()
    A_r = A(red)
    A_g = A(green)
    A_b = A(blue)
    
    noise_r = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    noise_g = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    noise_b = torch.rand(d,d, dtype=dt, device=dv)*noise_scale
    B_r = A_r + noise_r
    B_g = A_g + noise_g
    B_b = A_b + noise_b
    
    blurred_noised_img = np.zeros((d,d,3))
    blurred_noised_img[:,:,0] = normalize(B_r)
    blurred_noised_img[:,:,1] = normalize(B_g)
    blurred_noised_img[:,:,2] = normalize(B_b)
    # plt.imshow(blurred_noised_img)
    ax = axs0[1]
    ax.imshow(blurred_noised_img)
    ax.set_xticklabels([])
    ax.set_xlabel('Noisy Blurred Image', fontsize=fsize)
    
    if LSQR:
        img_LSQR, img_lifted_LSQR, tim_LSQR, average_iteration_LSQR = deblurring_solve(
            A, B_r, tb_g=B_g, tb_b =B_b, maxit=maxit, rtol=rtol, 
                            solver = lsqr)
        tt = 'LSQR ' + '(%.1f sec)' % (tim_LSQR)
        
        ax = axs0[2]
        ax.imshow(img_LSQR)
        ax.set_xticklabels([])
        ax.set_title(myMeasure(img_LSQR, original_img), fontsize=fsize)
        
        ax.set_xlabel(tt, fontsize=fsize)      
        ax.imshow(img_lifted_LSQR)
        
        print('LSQR', tim_LSQR, 'average iteration', average_iteration_LSQR)
        
    if LSMR:
        img_LSMR, img_lifted_LSMR, tim_LSMR, average_iteration_LSMR = deblurring_solve(
            A, B_r, tb_g=B_g, tb_b =B_b, maxit=maxit, rtol=rtol, 
                            solver = lsmr)
        tt = 'LSMR ' + '(%.1f sec)' % (tim_LSMR)
        
        ax = axs0[3]
        ax.imshow(img_LSMR)
        ax.set_xticklabels([])
        ax.set_title(myMeasure(img_LSMR, original_img), fontsize=fsize)
        
        ax.set_xlabel(tt, fontsize=fsize)      
        ax.imshow(img_lifted_LSMR)
        
        print('LSMR', tim_LSMR, 'average iteration', average_iteration_LSMR)
    fig0.savefig(os.path.join(filename, 'image'), dpi=mydpi)
    
    fig1, axs1 = plt.subplots(1, 4, constrained_layout=False, sharey=True, figsize=figsz1)
        
    if MR:
        img_MINRES, img_lifted_MINRES, tim_MINRES, average_iteration_MINRES = deblurring_solve(
            A, B_r, tb_g=B_g, tb_b =B_b, maxit=maxit, rtol=rtol, 
                            solver = myMINRES)
        tt = 'MINRES ' + '(%.1f sec)' % (tim_MINRES)
        
        ax = axs1[0]
        ax.set_xlabel(tt, fontsize=fsize) 
        ax.imshow(img_MINRES)
        ax.set_xticklabels([])
        ax.set_title(myMeasure(img_MINRES, original_img), fontsize=fsize)
        
        ax = axs1[1]
        ax.imshow(img_lifted_MINRES)
        ax.set_xticklabels([])
        ax.set_xlabel('Lifted MINRES', fontsize=fsize)      
        ax.set_title(myMeasure(img_lifted_MINRES, original_img), fontsize=fsize)
        print('MINRES', tim_MINRES, 'average iteration', average_iteration_MINRES)
    
    if MINRESQLP:
        img_MINRESQLP, img_lifted_MINRESQLP, tim_MINRESQLP, average_iteration_MINRESQLP = deblurring_solve(
            A, B_r, tb_g=B_g, tb_b =B_b, maxit=maxit, rtol=rtol, 
                            solver = MinresQLP)
        tt = 'MINRES-QLP ' + '(%.1f sec)' % (tim_MINRESQLP)
        
        ax = axs1[2]
        ax.set_xlabel(tt, fontsize=fsize)      
        ax.imshow(img_MINRESQLP)
        ax.set_xticklabels([])
        ax.set_title(myMeasure(img_MINRESQLP, original_img), fontsize=fsize)
        
        ax = axs1[3] 
        ax.imshow(img_lifted_MINRESQLP)
        ax.set_xticklabels([])
        ax.set_xlabel('Lifted MINRES-QLP', fontsize=fsize)  
        ax.set_title(myMeasure(img_lifted_MINRESQLP, original_img), fontsize=fsize)
        print('MINRES-QLP', tim_MINRESQLP, 'average iteration', average_iteration_MINRESQLP)
    
    
    fig1.savefig(os.path.join(filename, 'deblurring'), dpi=mydpi)
        
    #######################################################################    
    r = myA.rank
    d_tmp1 = [0.005, 0.01, 0.1, 1]
    d_tmp2 = [0.2, 0.5, 0.8, 1]
    
    if PMR:
        # index = np.random.choice(d, d, replace = False)  
        W = torch.randn(d, d, dtype=dt, device=dv)
        # W = W + W.T
        
        # Q1, R = house(myA.Toeplitz @ W, d)
        # Q2, R = house(W, d)
        # dC = C.shape[1]
        # index2 = np.random.choice(dC, dC, replace = False) 
        # C = C[:, index2]        
        # C = Q1    
           
        fig3, axs3 = plt.subplots(2, 4, constrained_layout=False, sharey=True, figsize=figsz2)
        for i in range(4):
            r_tmp = int(np.floor(r * np.sqrt(d_tmp2[i])))
            tim1 = time()
            Q2, R = house(W, r_tmp)
            C = Q2 @ np.diag(np.linspace(2, 1, r_tmp))
            tA = myA.kron_G(C)
            tb = myA.kron_G_vec(C)
            Sd = myA.kron_G_vec(C, re=True)
            tim2 = time() - tim1
            tim3 = time()
            img_RPMR, img_lifted_RPMR, tim_RPMR, average_iteration_RPMR = deblurring_solve(
                tA, tb(B_r), Sd=Sd, tb_g=tb(B_g), tb_b =tb(B_b), maxit=maxit, rtol=rtol)
            print('Range_preverved_PMINRES', tim_RPMR, 'average iteration', average_iteration_RPMR)
            tim4 = time() - tim3
            ax = axs3[0,i]
            ax.imshow(img_RPMR)
            if i == 0:
                ax.set_ylabel(r'$ Range(S_2) \not\subset Range(A) $', fontsize=fsize)
            ax.set_xticklabels([])
            ax.set_title(myMeasure(img_RPMR, original_img), fontsize=fsize)
            
            
            ax = axs3[1,i]
            ax.imshow(img_lifted_RPMR)
            if i == 0:
                ax.set_ylabel('Lifted of Above', fontsize=fsize)
            tt = 'Rank %s%% (%.1f/%.1f sec)' % (int(d_tmp2[i]*100), tim4, tim2)
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img_lifted_RPMR, original_img), fontsize=fsize)
            
        
        fig3.savefig(os.path.join(filename, 'PMR_deblurring'), dpi=mydpi)
    
        fig2, axs2 = plt.subplots(2, 4, constrained_layout=False, sharey=True, figsize=figsz2)
        for i in range(4):
            r_tmp = int(np.floor(r * np.sqrt(d_tmp1[i])))
            tim1 = time()
            Q1, R = house(myA.Toeplitz @ W, r_tmp)
            C = Q1 @ np.diag(np.linspace(2, 1, r_tmp))
            tA = myA.kron_G(C)
            tb = myA.kron_G_vec(C)
            Sd = myA.kron_G_vec(C, re=True)
            tim2 = time() - tim1
            tim3 = time()
            img_NPMR, img_lifted_NPMR, tim_NPMR, average_iteration_NPMR = deblurring_solve(
                tA, tb(B_r), Sd=Sd, tb_g=tb(B_g), tb_b =tb(B_b), maxit=maxit, rtol=rtol)
            print('Non_range_preverved_PMINRES', tim_NPMR, 'average iteration', average_iteration_NPMR)
            tim4 = time() - tim3
            ax = axs2[0,i]
            ax.imshow(img_NPMR)
            if i == 0:
                ax.set_ylabel(r'$ Range(S_1) \subset Range(A) $', fontsize=fsize)
            ax.set_xticklabels([])
            ax.set_title(myMeasure(img_NPMR, original_img), fontsize=fsize)
            
            ax = axs2[1,i]
            ax.imshow(img_lifted_NPMR)
            if i == 0:
                ax.set_ylabel('Lifted of Above', fontsize=fsize)
            if i < 1:
                tt = 'Rank %.1f%% (%.1f/%.1f sec)' % (d_tmp1[i]*100, tim4, tim2)
            else:
                tt = 'Rank %s%% (%.1f/%.1f sec)' % (int(d_tmp1[i]*100), tim4, tim2)
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img_lifted_NPMR, original_img), fontsize=fsize)
        
        fig2.savefig(os.path.join(filename, 'PMR_range_deblurring'), dpi=mydpi)
        
    #######################################################################        
    r = myA.rank
    d_tmp1 = [0.003, 0.006, 0.009, 0.012]
    if SVD:
        fig4, axs4 = plt.subplots(1, 4, constrained_layout=False, sharey=True, figsize=figsz1)
        for i in range(4):        
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
            tt = 'Rank %.1f%%' % (d_tmp1[i]*100)
            ax.set_xticklabels([])
            ax.set_xlabel(tt, fontsize=fsize)
            ax.set_title(myMeasure(img, original_img), fontsize=fsize)
            
        fig4.savefig(os.path.join(filename, 'Truncated_deblurring'), dpi=mydpi)
    
    
class Pblurring(object):
    def __init__(self, n, bandw, sigma, dtype=torch.float64, device="cpu"):
        # Note that Lambda = lambda * eye
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
        self.U, self.s, self.V = torch.svd(T1)
        self.time = time() - t0
    
    
    def range_(self, index):
        img = lambda X: self.U[:,index] @ X @ self.U[:,index].T
        return img
    
    def kron_(self):
        T = self.Toeplitz
        img = lambda X: T @ X @ T
        return img
    
    def A_(self):
        return kron(self.Toeplitz, self.Toeplitz).todense()
    
    def kron_G_vec(self, C, re=False):
        if re:
            img = lambda tX: C @ tX @ C.T # Sd @ tx
        else:
            img = lambda B: C.T @ B @ C # SdT @ b
        return img
    
    def kron_AG_vec(self, C, re=False):
        T = C.T @ self.Toeplitz
        if re:
            img = lambda tX: T.T @ tX @ T # Sd @ tx
        else:
            img = lambda B: T @ B @ T.T # SdT @ b
        return img
    
    def kron_G(self, C):
        T = self.Toeplitz
        T2 = C.T @ (T @ C)
        img = lambda X: T2 @ X @ T2
        return img
    
    def kron_AG(self, C):
        T = self.Toeplitz
        T2 = C.T @ (T @ (T @ (T @ C)))
        img = lambda X: T2 @ X @ T2
        return img

def plifting(x, r, br):
    return (x - (x@r)/(br@r) * br)

def lifting(x, r):
    return (x - (x@r)/(r@r) * r)

# def PSNR(img, img0):
#     # mse = (img - img0).norm(p='fro')
#     mse = torch.tensor(img - img0).norm('fro')
#     m, n, d = img.shape
#     return 20*torch.log10(np.sqrt(m*n*d)/mse)

# import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def myMeasure(img, img0):
    return 'PSNR: %.4s  SSIM: %s%%' % (psnr(img, img0).item(),int(100*ssim(img, 
                                       img0, channel_axis=-1, data_range=1).item()))

def TSVD(U, s, V, B_r, B_g, B_b, epsilon=0):
    sd = torch.zeros_like(s)
    index = s>epsilon
    s_tmp = s[index]
    sd[index] = 1/s_tmp
    Td = V @ torch.diag(sd) @ U.T
    d, d = B_r.shape
    img = np.zeros((d, d, 3))
    img[:,:,0] = normalize(Td @ B_r @ Td.T)
    img[:,:,1] = normalize(Td @ B_g @ Td.T)
    img[:,:,2] = normalize(Td @ B_b @ Td.T)
    return img
        
if __name__ == '__main__':
    main()
    