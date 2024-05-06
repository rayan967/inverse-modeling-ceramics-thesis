import sys
import os

import jcmwave
import numpy as np
import matplotlib.pyplot as plt

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from Optimization.optimizeaccuracy import *

from pathlib import Path


""" Test script for comparing shape derivatives between GPR and FEM  """

print("---- Loading data ----")
try:
    grid_coarse = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/Tests/grid_nOfPts_100.npy')
    scs_coarse  = np.load('D:/Uni/Zuse/simlopt/simlopt/Reconstruction/Tests/scs_nOfPts_100.npy')
except:
    print("Could not open the files...")


dim         = grid_coarse.shape[1]
m           = scs_coarse.shape[2]
nOfPts      = grid_coarse.shape[0]

Xt          = grid_coarse
yt          = (scs_coarse[:,:,0].T).reshape(-1,1)*1E9

epsXt       = 1E-6*np.ones((m,scs_coarse.shape[0]*scs_coarse.shape[1]))
eps         = epsXt[0,:]

""" Calculate HP """
hop = optimizeHP(Xt, None, yt, None, eps, None, gflag = 0 , toliter = 1E-7, maxiter = 300, random = 0, n=10)

""" Calculate derivatives via GPR """
sigma               = np.ones((1))
Ldf                 = np.ones([dim]);
hyperparametersdf   = np.concatenate([sigma,Ldf])

xrad = 0.4
xn   = 1.6
x    = np.array([[xrad,xn]])

# =============================================================================
# hyperparametersdf    = np.concatenate([sigma,Ldf])
# hyperparametersdf[0] = hop[0]
# hyperparametersdf[1] = hop[1]
# hyperparametersdf[2] = hop[2]
# =============================================================================

matricesdf          = kernelmatrices(x,Xt,hyperparametersdf,eps)
alpha               = np.linalg.inv(matricesdf[2]) @ yt
df                  = dGPR(x,Xt,matricesdf[1],hyperparametersdf)@alpha

mean                = matricesdf[1]@(np.linalg.solve(matricesdf[2],yt))

""" Plot solution """
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
plt.title("simple 3D scatter plot")

ax.scatter3D(Xt[:,0], Xt[:,1], yt, color = "green")
ax.scatter3D(x[:,0], x[:,1], mean, color = "red")
plt.show()

""" Calculate the derivative using FEM """
print ('Solving for lambda %5.2e' %(0.5,))
print ('Solving for radius %5.2e' %(xrad,))
print ('Solving for nglas  %5.2e' %(xn,))
print("\n")
keys = {'radius'   : xrad,
        'n_glass'  : xn,
        'lambda_0' : 0.5}
results = jcmwave.solve('mie2D.jcmp', keys=keys,
                        logfile=open(os.devnull, 'w'))

scs      = results[1]['ElectromagneticFieldEnergyFlux'][0][0].real

""" Derivatives at the new point - should be optional """
dEdrad = results[1]['d_rad']['ElectromagneticFieldEnergyFlux'][0][0].real
dEdrelperm = results[1]['d_relperm']['ElectromagneticFieldEnergyFlux'][0][0].real

"""Scale """
dEdrad = dEdrad*1E9
dEdrelperm = dEdrelperm*1E9

""" Difference """
diff = [df[0,0]-dEdrad, df[1,0]-dEdrelperm]
print("Difference in derivatives: {}".format( diff))

print("Difference in mean: {}".format( np.abs(mean[0,0]-scs*1E9)))