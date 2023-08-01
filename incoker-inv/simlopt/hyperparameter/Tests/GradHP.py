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
    res = np.zeros((2,X.shape[0]))
    for i in range(0,X.shape[0]):
        res[0,i] = np.sin(X[i,1])
        res[1,i] =  X[i,0] * np.cos(X[i,1])
    return  res

ranges  = np.array([[0,15],[0,15]])
ngrid   = [2,3,4,5,6,7,8,9,10]
errorbasic      = np.zeros((len(ngrid ),1))
errorbasicwopt  = np.zeros((len(ngrid ),1))
errorenhanced   = np.zeros((len(ngrid ),1))
errorgradenhanced = np.zeros((len(ngrid ),1))

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

    n  = 50
    x  = createPD(n, dim, "random", ranges)

    """ Calculate HP """
    hopgrad = optimizeHP(Xt, Xgrad, yt, ytg , epsXt, epsgradvector, gflag = 1 , toliter = 1E-8, maxiter = 400)
    print("\n")
    sigma               = np.ones((1))
    Ldf                 = np.ones([dim]);
    hyperparametersdf   = np.concatenate([sigma,Ldf])

    """ Hyperparameters for the gradient enhanced data """
    hgradopt            = np.concatenate([sigma,Ldf])
    hgradopt[0]         = hopgrad[0]
    hgradopt[1]         = hopgrad[1]
    hgradopt[2]         = hopgrad[2]

    """ Calculate the mean via enhanced GPR """
    ytilde          = np.concatenate((yt,ytg))
    mat             = kernelmatricesgrad(x,Xt,Xgrad,hyperparametersdf,epsXt,epsgradvector)
    meanenhanced    = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)

    """ Calculate the mean via enhanced GPR with optimized HP"""
    ytilde          = np.concatenate((yt,ytg))
    mat             = kernelmatricesgrad(x,Xt,Xgrad,hgradopt,epsXt,epsgradvector)
    meanegradnhanced = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)

    """ Analyticcaly """
    meananalytical    = fun(x)

    errorenhanced[i]      = 1/(ngrid[i]**2) * np.sum((meananalytical-meanenhanced)**2)
    errorgradenhanced[i]  = 1/(ngrid[i]**2) * np.sum((meananalytical-meanegradnhanced)**2)

""" Plot solution """
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax  = plt.gca()
ax.set_yscale('log')

plt.title("MSE over number of points")
plt.xlabel('Number of points')
plt.ylabel('Inf norm of error')


plt.plot((np.asarray(ngrid)**2).reshape(-1,1), errorenhanced , marker='*' , linestyle=':', color = "blue"   , label='Enhanced')
plt.plot((np.asarray(ngrid)**2).reshape(-1,1), errorgradenhanced , marker='s' , linestyle=':', color = "black"   , label='Enhanced optimized')
legend = ax.legend()
plt.grid(True)

plt.show()
