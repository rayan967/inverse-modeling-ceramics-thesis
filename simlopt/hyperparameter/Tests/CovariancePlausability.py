import sys,os

import numpy as np
import matplotlib.pyplot as plt
import jcmwave

from scipy.optimize import minimize

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.reconstruction.recon import*
from Training.training import *

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def fun(X):
    res = np.zeros((X.shape[0]))
    for i in range(0,X.shape[0]):
        res[i] = X[i,0] * np.sin(X[i,1])
    return  res

def gradf(X):
    res = np.zeros((2,X.shape[0]))
    for i in range(0,X.shape[0]):
        res[0,i] = np.sin(X[i,1])
        res[1,i] =  X[i,0] * np.cos(X[i,1])
    return  res

ranges  = np.array([[0,5],[0,5]])

dim     = 2
ngrid = 2
Xt      = createPD(ngrid, dim, "grid", ranges)
Xgrad   = Xt
yt      = fun(Xt)
ytg     = gradf(Xt)
ytg     = np.insert(ytg[1,:], np.arange(len(ytg[0,:])), ytg[0,:])
epsXt   = 1E-6*np.ones((1,ngrid**2))
epsgrad = 1E-6*np.eye((2*ngrid**2))
epsgradvector = 1E-6*np.ones((1,2*ngrid**2))

sigma               = np.ones((1))
L                   = np.array([2,1])

""" Build K """
N2 = Xt.shape[0]
N3 = Xgrad.shape[0]
D3 = Xgrad.shape[1]

""" Prescale data for the covaraince matrices only """
Xtscaled    = Xt/L
Xgradscaled = Xgrad/L


n2sq = np.sum(Xtscaled**2,axis=1);
n3sq = np.sum(Xgradscaled**2,axis=1);

# Kernel matrix Xt Xt
DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(Xtscaled,np.transpose(Xtscaled)))
KYY = sigma**2 * np.exp(-DYY / 2.0)
KYY = KYY

# Kernel matrix Xt Xgrad
DXtXgrad = np.transpose(np.outer(np.ones(N3),n2sq)) + np.outer(np.ones(N2),n3sq)-2* (np.dot(Xtscaled,np.transpose(Xgradscaled)))
kXtXgrad = sigma**2 * np.exp(-DXtXgrad / 2.0)
KXtXgrad = np.zeros((N2,N3*D3))
for i in range(0,N2):
    tmp = (Xt[i,:]-Xgrad)/L**2
    A = kXtXgrad[i,:]
    A = A[:,None] # Cast to coloumn vector
    tmp = np.multiply(tmp,A)
    res = np.reshape(tmp,(1,-1))
    KXtXgrad[i,:] = res


    # Kernel matrix Xgrad Xgrad
DXgradXgrad = np.transpose(np.outer(np.ones(N3),n3sq)) + np.outer(np.ones(N3),n3sq)-2* (np.dot(Xgradscaled,np.transpose(Xgradscaled)))
KXgXg       = sigma**2 * np.exp(-DXgradXgrad / 2.0)
# Second derivative
#Kfdy   = np.zeros((N3*D3,N3*D3));
tmprow = np.array([])
Kfdy = np.array([])
for i in range(0,N3):
    xi = Xgrad[i,:];
    for j in range(0,N3):
        xj = Xgrad[j,:];
        diff = np.outer(((xi-xj)/(L**2)),((xi-xj)/(L**2)))
        tmp = KXgXg[i,j]*( -diff + np.diag(1/L**2))
        if j == 0:
            tmprow = tmp
        else:
            tmprow = np.concatenate((tmprow,tmp),axis=1);
    if i == 0:
        Kfdy = tmprow
    else:
        Kfdy = np.concatenate((Kfdy,tmprow),axis=0);

Kfdy = Kfdy


""" Calculating the matrices by hand via Mathematica showed, that the implementation is correct - even with the scaling """