import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.deltap import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator

plt.close('all')

def freal(x):
    return np.sin(x)

def dfreal(x):
    #return x*np.cos(x)+np.sin(x)
    return np.cos(x)

""" Meta model """

ranges      = np.array([[-5,8]])
Xt          = createPD(9 , 1, "random", ranges)
Xgrad       = Xt
x           = createPD(1000, 1, "grid", ranges)


epsXt       = 1E-5*np.ones((1,Xt.shape[0]))
epsXtgrad   = 1E-5*np.ones((1,Xt.shape[0]))
dim         = Xt.shape[1]


yt          = freal(Xt)+epsXt.T
ygrad       = dfreal(Xgrad)

""" -------------------------- HP parameters -------------------------- """
bounds = ((1E-3,None),(1E-3,None))

""" ------------------------------------------ NON GRADIENT ------------------------------------------"""

""" Optimization """
H0          = []
sigma       = 1.
H0.append(sigma)
L               = [1.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0,bounds, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

""" Calculate the variances at xstar using Xtstar for every experiment 1....m"""
mat                 = kernelmatrices(x,Xt,HPm.T,epsXt)
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1]@np.linalg.inv(mat[2])@yt

""" Calculate derivatives at xstar using Xt and yt for every experiment 1....m"""
df         = np.zeros((1,1))
matdf      = kernelmatrices(x,Xt,HPm.T, epsXt)
dfgpr         = (dGPR(x,Xt,matdf[1],L)@(np.linalg.inv(matdf[2])@yt)).T


""" Calculate dp(x) """
dp = np.zeros((1,x.shape[0]))
dpgrad = np.zeros((1,x.shape[0]))
dprandom = np.zeros((1,x.shape[0]))
dpphys = np.zeros((1,x.shape[0]))

for i in range(x.shape[0]):
    dp[0,i] = -(1/( dfgpr[0,i]*dfgpr[0,i]*(1/variancevector[i])))*variancevector[i]*1/variancevector[i] * dfgpr[0,i]
    dpphys[0,i] = -(1/( dfgpr[0,i]*dfgpr[0,i]*(1/(variancevector[i]+0.5+0.001))))*(variancevector[i]+0.5+0.001)*1/(variancevector[i]+0.5+0.001) * dfgpr[0,i]
    dprandom[0,i] = -(1/( dfgpr[0,i]*dfgpr[0,i]*(1/(variancevector[i]+0.001))))*(variancevector[i]+0.001)*1/(variancevector[i]+0.001) * dfgpr[0,i]

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

for i in range(x.shape[0]):
    dpgrad[0,i] = -(1/( derivativesgrad[0,i]*derivativesgrad[0,i]*(1/variancevectorgrad[i])))*variancevectorgrad[i]*1/variancevectorgrad[i] * derivativesgrad[0,i]


""" REAL DATA """
df = dfreal(x)


""" Plotting """
fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(x, freal(x), 'r:', label=r'$f(x) = x\,\sin(x)$')

axs[0].scatter(Xt, yt,  label='Observations')
axs[0].scatter(Xgrad, yt,c = 'g', marker = "*", label='Observations')

axs[0].plot(x, y_pred, 'b-', label='Prediction')
axs[0].plot(x, y_predgrad, 'g--', label='Prediction')

axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(1000,1),
                        (y_pred + np.array([variancevector]).reshape(1000,1))[::-1]]),
         alpha=.25, fc='b', ec='None', label='variance interval')
axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_predgrad - np.array([variancevectorgrad]).reshape(1000,1),
                        (y_predgrad + np.array([variancevectorgrad]).reshape(1000,1))[::-1]]),
         alpha=.25, fc='g', ec='None', label='variance interval')

axs[0].grid(True)

axs[1].plot(x, dfgpr.T,'b-')
axs[1].plot(x, derivativesgrad.T,'g--')
axs[1].plot(x,df, 'r:')
axs[1].grid(True)
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$df$')

axs[2].plot(x, dp.T,'b-')
axs[2].plot(x, dpgrad.T,'r--')
# =============================================================================
# axs[2].plot(x, dpphys.T,'r-')
axs[2].plot(x, dprandom.T,'g-')
# =============================================================================
axs[2].set_ylim([-0.1,0.1])
axs[2].set_xlabel('$x$')
axs[2].set_ylabel('$dp$')

