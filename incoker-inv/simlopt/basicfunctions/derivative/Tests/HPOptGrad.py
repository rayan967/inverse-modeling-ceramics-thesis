import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from Optimization.optimizeaccuracy import *
from basicfunctions.reconstruction.deltap import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from HPOpt.utils.logmarginal import *
from HPOpt.utils.logmarginalgrad import *

plt.close('all')

def freal(x):
    return np.sin(x)

def dfreal(x):
    return np.cos(x)

""" Meta modell"""
xplot       = np.atleast_2d(np.linspace(0, 10, 1000)).T

xplot       = np.arange(-5, 5, 0.2).reshape(-1, 1)

ranges      = np.array([[-5,5]])
Xt          = createPD(6 , 1, "random", ranges)
#Xt = np.array([-4, -3, -2, -1, 1,3,4]).reshape(-1, 1)
x           = createPD(1000, 1, "grid", ranges)
Xgrad = Xt
ygrad = dfreal(Xgrad)
epsgrad = 1E-4*np.ones((1,Xt.shape[0]))
x           = np.arange(-5, 5, 0.2).reshape(-1, 1)
yt          = freal(Xt)
epsXt       = 1E-4*np.ones((1,Xt.shape[0]))
dim         = Xt.shape[1]


""" Optimization """
H0          = []
sigma       = 2.
H0.append(sigma)
L               = [10.]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

""" HP GRAD OPT """
region   = ((1e-4, 1E3), (1, 1E3))

almethod = 'SLSQP'
almethod = 'L-BFGS-B'

print("Performing hyperparameter optimization without gradient information")
print("Algorithm: {}".format(almethod))
optires   = minimize(logmarginallikelihoodgrad,
                     H0, args = (Xgrad,ygrad,epsgrad),
                     method=almethod,
                     bounds = region,
                     tol = 1E-7, options={'maxiter': 300, 'disp':True})
if optires.success == True:
        print("Hyperparameter optimization succeded in: {} iterations".format(optires.nit))
        print("    Hyperparameter: {}".format(np.abs(optires.x)))

        sigma           = optires.x[0]
        L               = optires.x[1:len(Hopt)]
        HPgrad          = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)
else:
        print("Hyperparameter optimization failed")


""" Calculate the variances at xstar using Xtstar for every experiment 1....m"""
mat                 = kernelmatrices(x,Xt,HPm.T,epsXt)
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1]@np.linalg.inv(mat[2])@yt

""" Calculate derivatives at xstar using Xt and yt for every experiment 1....m"""
sigmaU              = np.array([1])
LU                  = np.ones([1])
derivatives         = np.zeros((1,1))
hyperparametersdf   = np.concatenate([np.array([sigmaU]) , np.array([LU]) ])
matdf               = kernelmatrices(x,Xt,HPgrad.T, epsXt)
#x = np.array([[0.3]])
derivatives         = (dGPR(x,Xt,matdf[1],L)@(np.linalg.inv(matdf[2])@yt)).T
df = dfreal(x)

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(x, freal(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
axs[0].scatter(Xt, yt,  label='Observations')
axs[0].plot(x, y_pred, 'b-', label='Prediction')

axs[0].fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(50,1),
                        (y_pred + np.array([variancevector]).reshape(50,1))[::-1]]),
         alpha=.5, fc='b', ec='None', label='variance interval')
axs[0].grid(True)

axs[1].plot(x, derivatives.T)
axs[1].plot(x,df, 'r:')
axs[1].grid(True)
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$1/df$')




