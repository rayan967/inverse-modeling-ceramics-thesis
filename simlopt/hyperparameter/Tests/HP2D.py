import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator



import scipy.stats as stats
from scipy.optimize import check_grad
import math

plt.close('all')


def fun(X):

    res = np.zeros((X.shape[0]))

    for i in range(0,X.shape[0]):
        #res[i] = X[i,0] * np.sin(X[i,1])
        res[i] = X[i,0]**2+ X[i,1]**2

    return  res

def gradf(X):

    res = []

    for i in range(0,X.shape[0]):
        dfx1 = 2*X[i,0]
        dfx2 = 2*X[i,1]
        res.append(dfx1)
        res.append(dfx2)
    return  res


""" Meta modell"""

ranges      = np.array([[0,1],[0,1]])
Xt          = createPD(20 , 2, "grid", ranges)
# =============================================================================
# Xt = np.array(([[0,0],
#                 [0.5,0],
#                 [1,0],
#                 [0,0.5],
#                 [1,0.5],
#                 [0,1],
#                 [0.5,1],
#                 [1,1]]))
# =============================================================================
Xgrad       = Xt
x           = createPD(5, 2, "random", ranges)
#x =  np.array(([[0.5,0.5]]))

yt          = fun(Xt)
ygrad       = gradf(x)

epsXt       = 1E-5*np.ones((1,Xt.shape[0]))
#epsXtgrad   = 1E-5*np.ones((1,Xt.shape[0]))
dim         = Xt.shape[1]

""" -------------------------- HP parameters -------------------------- """
bounds = ((1E-3,None),(1E-3,None),(1E-3,None))

""" ------------------------------------------ NON GRADIENT ------------------------------------------"""

""" Optimization """
H0          = []
sigma       = 1.
H0.append(sigma)
L               = [1.,1.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0,bounds, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]

#sigma = 1
#L = np.array([1.,1.])
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

mat                 = kernelmatrices(x,Xt,HPm[0,:],epsXt)

""" Calculate derivatives at xstar using Xt and yt for every experiment 1....m"""
sigmaU              = np.array([1])
LU                  = np.ones([1])
derivatives         = np.zeros((x.shape[0]*2,1))
derivatives         = (dGPR(x,Xt,mat[1],L)@(np.linalg.inv(mat[2])@yt)).T
differror           = np.abs(np.asarray(ygrad) -derivatives)


""" ------------------------------------------ WITH GRADIENT DATA ------------------------------------------ """
H0          = []
sigma       = 1.
H0.append(sigma)
L               = [1.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,Xgrad,yt,ygrad,epsXt,epsXtgrad,H0,bounds, 1)
sigma           = np.log(Hopt[0])
L               = Hopt[1:len(Hopt)]
HPgrad          = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

ytilde              = np.concatenate((yt,ygrad))
mat                 = kernelmatricesgrad(x,Xt,Xgrad,HPgrad.T,epsXt,epsXtgrad)
variancevectorgrad  = np.diag(mat[0] -np.concatenate((mat[1],mat[3]),axis = 1)@np.linalg.inv(mat[5])@(np.concatenate((mat[1],mat[3]),axis = 1)).T)
y_predgrad          = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)
derivativesgrad     = (dGPRgrad(x,Xt,Xgrad,sigma,L) @ (np.linalg.inv(mat[5]) @ ytilde)).T

""" REAL DATA """
df = dfreal(x)


""" Plotting """
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(x, freal(x), 'r:', label=r'$f(x) = x\,\sin(x)$')

axs[0].scatter(Xt, yt,  label='Observations')
axs[0].scatter(Xgrad, yt,c = 'g', marker = "*", label='Observations')

axs[0].plot(x, y_pred, 'b-', label='Prediction')
axs[0].plot(x, y_predgrad, 'g--', label='Prediction')

axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(100,1),
                        (y_pred + np.array([variancevector]).reshape(100,1))[::-1]]),
         alpha=.25, fc='b', ec='None', label='variance interval')
axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_predgrad - np.array([variancevectorgrad]).reshape(100,1),
                        (y_predgrad + np.array([variancevectorgrad]).reshape(100,1))[::-1]]),
         alpha=.25, fc='g', ec='None', label='variance interval')

axs[0].grid(True)

axs[1].plot(x, derivatives.T,'b-')
axs[1].plot(x, derivativesgrad.T,'g--')
axs[1].plot(x,df, 'r:')
axs[1].grid(True)
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$1/df$')




