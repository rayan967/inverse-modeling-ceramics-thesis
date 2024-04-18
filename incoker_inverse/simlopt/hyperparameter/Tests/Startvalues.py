import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.deltap import *

print("---- Loading data ----")
try:
    data = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/data/data_36.npy')
    data = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/data/data3D.npy')
    print("Data done...")
    freal = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/data/freal.npy')
    freal = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/data/freal3D.npy')
except:
    print("Could not open the files...")

dim    = 3
m      = data.shape[2]

""" Extract data """
Xt = data[:,0:3,:]
yt = data[:,3,:]

""" Extract gradient data """
Xgrad = data[:,0:3,:]
yg    = data[:,4:7,:]

""" Get sub grad data for testing purposes"""
# =============================================================================
# Xgrad = Xgrad[3:7,:,:]
# yg    = yg[3:7,:,:]
# =============================================================================

""" Reshape gradient data from tensor to matrix """
# =============================================================================
# for i in range(0,m-1):
#     if i == 0:
#         Xt = grid_coarse
#     Xt  = np.dstack((Xt,grid_coarse))
#
# for i in range(0,m-1):
#     res = (scs_coarse[:,:,i]).reshape(-1,1)
#     if i == 0:
#         yt = res
#     yt  = np.hstack((yt,res))
# =============================================================================

ygrad = np.zeros((yg.shape[0]*dim, m))
for i in range(0,m-1):
    ygrad[:,i] = np.insert(yg[:,:,i],1,[])


""" Final data strcture
Xt      = n x dim x m
yt      = n x m
Xgrad   = n x dim x m
ygrad   = n*dim x m
"""

""" -------------------------- Prescale data -------------------------- """
scalefactor = 1E9
yt          = yt    * scalefactor
ygrad       = ygrad * scalefactor
freal       = freal * scalefactor



""" -------------------------- Data error ----------------------------- """
epsXt    = 1E-5*np.ones((m,Xt.shape[0]))
epsXgrad = 1E-5*np.ones((m,Xgrad.shape[0]*dim))

ygrad = np.array([])
Xgrad = np.array([])

""" -------------------------- Training parameters -------------------- """
itermax     = 1
nofdatapts  = Xt.shape[0]
rho         = 0.5
nofsamples  = nofdatapts / rho
minimaldist = 1E-4
Wbudget     = 100000
threshold   = 1E-6
eps0        = list(np.repeat(1E-5, m))

""" ----------------------- Data error ----------------------- """
n           = Xt.shape[0]
ngrad       = Xgrad.shape[0]

dim         = Xt.shape[1]
m           = Xt.shape[2]
scalefactor = 1E9

print("---- Initial parameters ----")
print("Problem dimension:                {}".format(dim))
print("Number of experiments:            {}".format(m))
print("Number of training points:        {}".format(n))
print("Number of ge-training points:     {}".format(n))
print('Scale factor:                     %5.2e' %(scalefactor))

""" ----------------------- Initial Hyperparameter optimization ------------------------"""
"""
For the given training data / start distribution a first hyperparameter optimization is
performed using an anisotropic RBF kernel.
Since we dont expect the data to differ much between the different the optimization is done for the first
set of data / experiment.
"""
print("\n")
print("---- Initial Hyperparameter Optimization ----")

HPm = np.zeros((m,1+dim))

"""

      | simga_1^1 L_1^1 .... L_d^1  |
Hpm = | ....                        |
      | simga_1^1 L_1^1 .... L_d^1  |

"""

""" Automatic start value """
H0      = []
sigma   = np.ceil(np.abs(np.max(yt)-np.min(yt)))
H0.append(sigma)

dim     = Xt.shape[1]
res     = np.sign(Xt)*np.ceil(np.abs(Xt))
sigma   = 10
L       = [0.1 , 0.1 , 0.1]
H0 = H0+L

for i in range(0,m):
    if Xgrad.size != 0:
        print("Metamodel: {}".format(i))
        gflag       = 1
        Hopt        = optimizeHP(Xt[:,:,i],Xgrad[:,:,i],yt[:,i],ygrad[:,i],epsXt[i,:],epsXgrad[i,:],H0, gflag)
        sigma       = Hopt[0]
        L           = Hopt[1:len(Hopt)]
        HPm[i,:]    = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)
        H0          = HPm[i,:]
        print("\n")

    else:
        print("Metamodel: {}".format(i))
        gflag           = 0
        Hopt            = optimizeHP(Xt[:,:,i],None,yt[:,i],None,epsXt[i,:],None,H0, gflag)
        sigma           = Hopt[0]
        L               = Hopt[1:len(Hopt)]
        HPm[i,:]        = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)
        H0              = HPm[i,:]
        print("\n")