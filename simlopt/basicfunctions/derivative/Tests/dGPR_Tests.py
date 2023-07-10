""" Script for testing the estimation of the gradient used in GPR

"""
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

    res = np.zeros((X.shape[0],2))
    for i in range(0,X.shape[0]):
        res[i,0] = np.sin(X[i,1])
        res[i,1] =  X[i,0] * np.cos(X[i,1])
    return  res

ranges = np.array([[0,10],[0,5]])
ngrid   = 10
dim     = 2
Xt      = createPD(ngrid, dim, "grid", ranges)
yt      = fun(Xt)
#yt      = np.sum(Xt**2,axis = 1)
epsXt   = 1E-6*np.ones((1,ngrid**2))


""" Calculate HP """
#hop = optimizeHP(Xt, None, yt, None, epsXt, None, gflag = 0 , toliter = 1E-7, maxiter = 300, random = 0, n=10)

""" Calculate derivatives via GPR """
sigma               = np.ones((1))
Ldf                 = np.ones([dim]);
hyperparametersdf   = np.concatenate([sigma,Ldf])
# =============================================================================
#
# hyperparametersdf    = np.concatenate([sigma,Ldf])
# hyperparametersdf[0] = hop[0]
# hyperparametersdf[1] = hop[1]
# hyperparametersdf[2] = hop[2]
# =============================================================================

n               = 1
x               = createPD(n, dim, "random", ranges)
matricesdf      = kernelmatrices(x,Xt,hyperparametersdf,epsXt)
alpha           = np.linalg.inv(matricesdf[2]) @ yt
df              = dGPR(x,Xt,matricesdf[1],Ldf)@alpha
gradf           = gradf(x)


