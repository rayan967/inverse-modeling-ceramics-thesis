import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator

plt.close('all')

def freal(x):
    return np.sin(x)

def dfreal(x):
    #return x*np.cos(x)+np.sin(x)
    return np.cos(x)

""" Meta modell"""

ranges      = np.array([[-5,8]])
Xt          = createPD(5 , 1, "random", ranges)
Xt = np.expand_dims(Xt, axis=2)
Xgrad       = Xt
x           = createPD(100, 1, "grid", ranges)

yt          = freal(Xt[:,:,0])
ygrad       = dfreal(Xgrad)
epsXt       = 1E-5*np.ones((1,Xt.shape[0]))
epsXtgrad   = 1E-5*np.ones((1,Xt.shape[0]))
dim         = Xt.shape[1]

""" -------------------------- HP parameters -------------------------- """
region = ((1E-3,None),(1E-3,None))

""" ------------------------------------------ NON GRADIENT ------------------------------------------"""

""" Optimization """
# =============================================================================
# H0          = []
# sigma       = 1.
# H0.append(sigma)
# L               = [1.]
# H0              = H0+L
# gflag           = 0
# Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0,bounds, gflag)
# sigma           = Hopt[0]
# L               = Hopt[1:len(Hopt)]
# HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)
# =============================================================================
HPm = np.zeros((1, dim + 1))
HPm[0, :] = optimizehyperparameters(Xt, None, yt, None, epsXt, None, region, "unit")
""" Calculate the variances at xstar using Xtstar for every experiment 1....m"""
mat                 = kernelmatrices(x,Xt[:,:,0],HPm[0,:],epsXt[0,:])
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1]@np.linalg.inv(mat[2])@yt

# =============================================================================
# """ Calculate derivatives at xstar using Xt and yt for every experiment 1....m"""
# derivatives         = np.zeros((1,1))
# matdf               = kernelmatrices(x,Xt,HPm.T, epsXt)
# derivatives         = (dGPR(x,Xt,matdf[1],L)@(np.linalg.inv(matdf[2])@yt)).T
# 
# =============================================================================


# =============================================================================
# """ ------------------------------------------ WITH GRADIENT DATA ------------------------------------------ """
# H0          = []
# sigma       = 1.
# H0.append(sigma)
# L               = [1.]
# H0              = H0+L
# gflag           = 0
# Hopt            = optimizeHP(Xt,Xgrad,yt,ygrad,epsXt,epsXtgrad,H0,bounds, 1)
# sigma           = np.log(Hopt[0])
# L               = Hopt[1:len(Hopt)]
# HPgrad          = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)
# 
# ytilde              = np.concatenate((yt,ygrad))
# mat                 = kernelmatricesgrad(x,Xt,Xgrad,HPgrad.T,epsXt,epsXtgrad)
# variancevectorgrad  = np.diag(mat[0] -np.concatenate((mat[1],mat[3]),axis = 1)@np.linalg.inv(mat[5])@(np.concatenate((mat[1],mat[3]),axis = 1)).T)
# y_predgrad          = np.concatenate((mat[1],mat[3]),axis = 1) @ (np.linalg.inv(mat[5]) @ ytilde)
# derivativesgrad     = (dGPRgrad(x,Xt,Xgrad,sigma,L) @ (np.linalg.inv(mat[5]) @ ytilde)).T
# 
# """ REAL DATA """
# df = dfreal(x)
# 
# 
# =============================================================================
""" Plotting """
fig, axs = plt.subplots(2)
fig.suptitle('Compare non-grad and grad')

axs[0].plot(x, freal(x), 'r:', label=r'$f(x) = x\,\sin(x)$')

axs[0].scatter(Xt, yt,  label='Observations')
axs[0].scatter(Xgrad, yt,c = 'g', marker = "*", label='Observations')

axs[0].plot(x, y_pred, 'b-', label='Prediction')
#axs[0].plot(x, y_predgrad, 'g--', label='Prediction')

axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(100,1),
                        (y_pred + np.array([variancevector]).reshape(100,1))[::-1]]),
         alpha=.25, fc='b', ec='None', label='variance interval')
# =============================================================================
# axs[0].fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_predgrad - np.array([variancevectorgrad]).reshape(100,1),
#                         (y_predgrad + np.array([variancevectorgrad]).reshape(100,1))[::-1]]),
#          alpha=.25, fc='g', ec='None', label=
#          'variance interval')
# =============================================================================
axs[0].set(xlabel='x',ylabel='y(x)')
axs[0].axes.xaxis.set_ticklabels([])
axs[0].axes.yaxis.set_ticklabels([])
# =============================================================================
# axs[0].tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# axs[0].tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# =============================================================================
axs[0].grid(True)
# =============================================================================
# 
# axs[1].plot(x, derivatives.T,'b-')
# axs[1].plot(x, derivativesgrad.T,'g--')
# axs[1].plot(x,df, 'r:')
# axs[1].grid(True)
# axs[1].set_xlabel('$x$')
# axs[1].set_ylabel('$1/df$')
# 
# 
# =============================================================================


