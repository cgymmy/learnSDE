import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy
from scipy import sparse
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

def Get_Ele_Info(h, p, q, f, ne):
    Kks=np.zeros((ne,2,2));    
    Kks[:,0,0]=p/h; Kks[:,0,1]=-p/h; Kks[:,1,0]=-p/h; Kks[:,1,1]=p/h;
    Mks=np.zeros_like(Kks)
    Mks[:,0,0]=q*h/3; Mks[:,0,1]=q*h/6; Mks[:,1,0]=q*h/6; Mks[:,1,1]=q*h/3
    bks=np.zeros((ne,2))
    bks[:,0]=f*(h / 2); bks[:,1]=f*(h / 2)
    return Kks, Mks, bks


def FEM_Solver1D_r1(ne, p, q, f):
    h = 1 / ne
    x = np.linspace(0, 1, ne + 1)
    nvtx = ne + 1
    Kks,Mks,bks=Get_Ele_Info(h,p,q,f,ne)
    elt2vert=np.vstack((np.arange(0, nvtx - 1, dtype='int'),
                        np.arange(1, nvtx, dtype='int')))
    b = np.zeros(nvtx)
    K = sum(sparse.csc_matrix((Kks[:,row_no,col_no],(elt2vert[row_no,:],elt2vert[col_no,:])),
                               (nvtx,nvtx))
              for row_no in range(2)  for col_no in range(2))
    M = sum(sparse.csc_matrix((Mks[:,row_no,col_no],(elt2vert[row_no,:],elt2vert[col_no,:])),
                               (nvtx,nvtx))
              for row_no in range(2)  for col_no in range(2))
    for row_no in range(2):
        nrow=elt2vert[row_no,:]
        b[nrow]=b[nrow]+bks[:,row_no]
    A=K + M
    # impose homogeneous boundary condition
    A=A[1:-1,1:-1]; K=K[1:-1,1:-1]; M=M[1:-1,1:-1]
    b=b[1:-1]
    # solve linear system for interior degrees of freedom
    u_int=sparse.linalg.spsolve(A,b)
    # add in boundary data 
    uh=np.hstack([0,u_int,0])
    # plotting commands removed 
    return uh,A,b,K,M


def Uniform_Mesh(ns):
    n = ns + 1
    h = 1/ns
    x = np.linspace(0, 1, ns+1)
    y = np.linspace(0, 1, ns+1)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()
    yv = yv.ravel()
    nvtx = n**2
    nsqr = ns**2
    ne = 2*nsqr
    elt2vert = np.zeros((ne, 3), dtype='int')
    vv=np.reshape(np.arange(0, nvtx),(ns + 1,ns + 1), order='F')
    v1=vv[:-1, :-1]; v2=vv[1:, :-1]; v3=vv[:-1, 1:]; v4=vv[1:, 1:]
    elt2vert[:nsqr, :] = np.vstack((v1.ravel(), v2.ravel(), v3.ravel())).T
    elt2vert[nsqr:, :] = np.vstack((v4.ravel(), v3.ravel(), v2.ravel())).T
    # plt.figure(figsize=(8, 6)) 
    # plt.axis('equal')
    # plt.triplot(xv.ravel(),yv.ravel(), elt2vert, 'k-')
    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    return h, xv, yv, elt2vert, nvtx, ne

def Get_Jacobian_Info(xv, yv, ne, elt2vert):
    Jks = np.zeros((ne, 2, 2))
    invJks = np.zeros((ne, 2, 2))
    x1=xv[elt2vert[:,0]]; x2=xv[elt2vert[:,1]]; x3=xv[elt2vert[:,2]]
    y1=yv[elt2vert[:,0]]; y2=yv[elt2vert[:,1]]; y3=yv[elt2vert[:,2]]
    Jks[:,0,0]=x2 - x1; Jks[:,0,1]=y2 - y1
    Jks[:,1,0]=x3 - x1; Jks[:,1,1]=y3 - y1
    detJks=Jks[:,0,0]*Jks[:,1,1]-Jks[:,0,1]*Jks[:,1,0]
    invJks[:,0,0]=(y3 - y1)/detJks
    invJks[:,0,1]=(y1 - y2)/detJks
    invJks[:,1,0]=(x1 - x3)/detJks
    invJks[:,1,1]=(x2 - x1)/detJks
    return Jks, invJks, detJks

def Get_Integration_Info_r1(ne, invJks, detJks, a, f):
    Ak = np.zeros((ne, 3, 3))
    bk = np.zeros((ne, 3))
    dpsi_ds=np.array([-1,1,0])
    dpsi_dt=np.array([-1,0,1])
    for i in range(3):
        for j in range(3):
            grad=np.array([[dpsi_ds[i],dpsi_ds[j]],
                           [dpsi_dt[i],dpsi_dt[j]]])       
            v1=np.dot(invJks[:,:,:], grad[:, 0])
            v2=np.dot(invJks[:,:,:], grad[:, 1])
            tmp = detJks * np.sum(v1 * v2, axis=1)
            Ak[:,i,j] = Ak[:,i,j] + a*tmp/ 2
        bk[:,i]=bk[:,i] + f*detJks / 6
    return Ak,bk

def FEM_Solver2D_r1(ns, xv, yv, elt2vert, nvtx, ne, a, f):
    Jks, invJks, detJks = Get_Jacobian_Info(xv, yv, ne, elt2vert)
    Aks, bks = Get_Integration_Info_r1(ne, invJks,detJks,a,f)
    A = sparse.csc_matrix((nvtx,nvtx))
    A = sum(sparse.csc_matrix((Aks[:,row_no,col_no], (elt2vert[:,row_no], elt2vert[:,col_no])), (nvtx,nvtx)) for row_no in range(3)  for col_no in range(3))
    b=np.zeros(nvtx)
    for row_no in range(3):
        nrow=elt2vert[:,row_no]
        b[nrow] = b[nrow]+bks[:,row_no]
    # get discrete Dirichlet boundary data 
    b_nodes = np.where((xv == 0) | (xv == 1) | (yv == 0) | (yv == 1))[0]
    int_nodes = np.ones(nvtx, dtype='bool');    
    int_nodes[b_nodes]=False;    
    b_int=np.squeeze(b[int_nodes])
    # print(b_int)
    wB=g_eval(xv[b_nodes], yv[b_nodes])
    # solve linear system for interior nodes;
    Ab = A[int_nodes,:];    Ab = Ab[:,b_nodes]
    rhs=b_int - Ab.dot(wB)
    A_int=A[int_nodes,:];    A_int=A_int[:,int_nodes]
    u_int=sparse.linalg.spsolve(A_int,rhs)
    # combine with boundary data and plot
    uh=np.zeros(nvtx)
    uh[int_nodes] = u_int
    uh[b_nodes] = wB
    return uh, u_int, A_int, rhs

def g_eval(x,y):
    g=np.zeros(x.shape)
    return g      