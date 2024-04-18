""" Script for testing the estimation of the gradient used in GPR

"""
import sys,os

import numpy as np
import matplotlib.pyplot as plt

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
ngrid   = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
errorbasic = np.zeros((len(ngrid ),1))
errorenhanced = np.zeros((len(ngrid ),1))

for i in range(0,len(ngrid )):
    dim     = 2
    Xt      = createPD(ngrid[i], dim, "grid", ranges)
    Xgrad   = Xt
    yt      = fun(Xt)
    ytg     = gradf(Xt)
    ytg     = np.insert(ytg[1,:], np.arange(len(ytg[0,:])), ytg[0,:])
    epsXt   = 1E-6*np.ones((1,ngrid[i]**2))
    epsgrad = 1E-6*np.eye((2*ngrid[i]**2))
    epsgradvector = 1E-6*np.ones((1,2*ngrid[i]**2))

    """ Calculate HP """
    #hop = optimizeHP(Xt, None, yt, None, epsXt, None, gflag = 0 , toliter = 1E-7, maxiter = 300, random = 0, n=10)
    sigma               = np.ones((1))
    Ldf                 = np.ones([dim]);
    hyperparametersdf   = np.concatenate([sigma,Ldf])

    n                   = 10
    x                   = createPD(n, dim, "random", ranges)
    x                   = np.array([[5,2.5]])

    """ Calculate derivatives via basic GPR """
    matricesdf      = kernelmatrices(x,Xt,hyperparametersdf,epsXt)
    alpha           = np.linalg.inv(matricesdf[2]) @ yt
    dfbasic         = dGPR(x,Xt,matricesdf[1],Ldf)@alpha

    """ Calculate derivatives via enhanced GPR """
    ytilde          = np.concatenate((yt,ytg))
    mat             = kernelmatricesgrad(x,Xt,Xgrad,hyperparametersdf,epsXt,epsgradvector)
    dfenhanced      = (dGPRgrad(x, Xt, Xgrad) @ (np.linalg.inv(mat[5]) @ ytilde)).T

    """ Analyticcaly """
    dfanalytical    = gradf(x)


    errorbasic[i]      = np.linalg.norm( ( dfanalytical[0,0] -dfbasic[0],dfanalytical[1,0] -dfbasic[1]) ,np.inf)
    errorenhanced[i]   = np.linalg.norm( ( dfanalytical[0,0] -dfenhanced[0],dfanalytical[1,0] -dfenhanced[1]) ,np.inf)


""" Plot solution """
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax  = plt.gca()
ax.set_yscale('log')

plt.title("Inf norm over number of points")
plt.xlabel('Number of points')
plt.ylabel('Inf norm of error')

plt.plot((np.asarray(ngrid)**2).reshape(-1,1), errorbasic    , marker='*' , linestyle=':', color = "green" , label='Basic')
plt.plot((np.asarray(ngrid)**2).reshape(-1,1), errorenhanced , marker='o' , linestyle=':', color = "red"   , label='Enhanced')
legend = ax.legend()
plt.grid(True)

plt.show()
