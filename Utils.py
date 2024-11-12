import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy
# BCCB
def Circulant_Sample_2d(C_red, n1, n2, seed = 24):
    N = n1 * n2
    Lam = N * np.fft.ifft2(C_red)
    d = np.ravel(np.real(Lam))
    d_minus = np.maximum(- d, 0)
    if np.max(d_minus > 0):
        print(f'rho(D_minus)={np.max(d_minus)}')
    np.random.seed(seed)
    xi=np.random.randn(n1,n2) + 1j*np.random.randn(n1,n2)
    V=(Lam ** 0.5)*xi
    Z = np.fft.fft2(V) / sqrt(N)
    X=np.real(Z);    Y=np.imag(Z)
    return X, Y

def Reduced_Cov(n1, n2, dx1, dx2, c):
    C_red = np.zeros((2*n1 - 1, 2*n2 - 1))
    for i in range(2*n1 - 1):
        for j in range(2*n2 - 1):
            C_red[i, j] = c((i+1-n1)*dx1, (j+1-n2)*dx2)
    return C_red

def Circulant_Embed_Sample_2d(C_red, n1, n2, seed = 24):
    N = n1 * n2
    tilde_C_red = np.zeros((2 * n1, 2 * n2))
    tilde_C_red[1:2*n1, 1:2*n2] = C_red
    tilde_C_red = np.fft.fftshift(tilde_C_red)
    u1, u2 = Circulant_Sample_2d(tilde_C_red, 2*n1, 2*n2, seed)
    u1 = np.ravel(u1);  u2=np.ravel(u2)
    u1 = u1[0:2*N]; u1 = u1.reshape((n1,2 * n2)); u1 = u1[:,::2]
    u2 = u2[0:2*N]; u2 = u2.reshape((n1,2 * n2)); u2 = u2[:,::2]
    return u1, u2

def Circulant_Embed_Approx_2d(C_red,n1,n2,m1,m2, seed = 24):
    nn1 = n1 + m1;    nn2 = n2 + m2
    N=nn1 * nn2
    tilde_C_red=np.zeros((2 * nn1,2 * nn2))
    tilde_C_red[1:2 * nn1,1:2 * nn2]=C_red
    tilde_C_red=np.fft.fftshift(tilde_C_red)
    u1,u2=Circulant_Sample_2d(tilde_C_red, 2 * nn1, 2 * nn2, seed)
    # print(u1.shape, u2.shape)
    u1=np.ravel(u1);    u2=np.ravel(u2)
    u1=u1[0:2 * N];    u1=u1.reshape((nn1,2 * nn2));    u1=u1[0:n1, 0:2*n2:2]
    u2=u2[0:2 * N];    u2=u2.reshape((nn1,2 * nn2));    u2=u2[0:n1, 0:2*n2:2]
    return u1,u2

def sep_exp(x1,x2,ell_1,ell_2):
    c=exp(- abs(x1) / ell_1 - abs(x2) / ell_2)
    return c

def gaussA_exp(x1,x2,a11,a22,a12):
    c=exp(- ((x1 ** 2 * a11 + x2 ** 2 * a22) - 2 * x1 * x2 * a12))
    return c