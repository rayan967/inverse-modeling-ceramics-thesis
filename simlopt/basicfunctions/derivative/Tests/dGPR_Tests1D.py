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

    return  np.sin(X)

def gradf(X):

    return  np.cos(X)


Xt      =  np.array([[0],[0.5],[1],[2],[3]])
yt      = fun(Xt)
#yt      = np.sum(Xt**2,axis = 1)
epsXt   = 1E-6*np.ones((1,5))


""" Calculate HP """
#hop = optimizeHP(Xt, None, yt, None, epsXt, None, gflag = 0 , toliter = 1E-7, maxiter = 300, random = 0, n=10)

""" Calculate derivatives via GPR """
# =============================================================================
# sigma               = np.ones((1))
# Ldf                 = np.ones([1]);
# hyperparametersdf   = np.concatenate([sigma,Ldf])
# # =============================================================================
# #
# # hyperparametersdf    = np.concatenate([sigma,Ldf])
# # hyperparametersdf[0] = hop[0]
# # hyperparametersdf[1] = hop[1]
# # hyperparametersdf[2] = hop[2]
# # =============================================================================
# =============================================================================
H0          = []
sigma       = 1.
H0.append(sigma)
L               = [1.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

n               = 1
x               = np.array([[1]])
matricesdf      = kernelmatrices(x,Xt,HPm.T,epsXt)
y_pred          = matricesdf[1]@np.linalg.inv(matricesdf[2])@yt

h = 1E-5
xh               = np.array([[1+h]])
matricesdf      = kernelmatrices(xh,Xt,HPm.T,epsXt)
y_predh         = matricesdf[1]@np.linalg.inv(matricesdf[2])@yt

dffd = (y_predh-y_pred)/h

""" real """
yreal = fun(x)


alpha           = np.linalg.inv(matricesdf[2]) @ yt
tmp = dGPR(x,Xt,matricesdf[1],L)
df              = dGPR(x,Xt,matricesdf[1],L)@alpha
gradf           = gradf(x)


