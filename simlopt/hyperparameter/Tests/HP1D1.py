import matplotlib.pyplot as plt
import numpy as np
from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.deltap import *


plt.close('all')

def freal(x):
    return x*np.sin(x)

def dfreal(x):
    return x*np.cos(x)+np.sin(x)

""" Meta modell"""
xplot       = np.atleast_2d(np.linspace(0, 10, 1000)).T
ranges      = np.array([[-2.0,10]])
#Xt          = createPD(7 , 1, "random", ranges)

Xt= np.array([[1.76697],
[3.08641],
[7.54391],
[3.3819],
[3.06467],
[0.684323],
[6.28115]])

a = 1E0
b = 1E-2
c = 1E-1

x           = createPD(1000, 1, "grid", ranges)


err         = 1E0
epsXt       = err*np.ones((1,Xt.shape[0]))
epsXt       = np.array([[a],[a],[b],[c],[a],[b],[c]]).T
dim         = Xt.shape[1]

yt          = freal(Xt)+epsXt.T

bounds = ((1E-3,None),(1E-3,None))

H0          = []
sigma       = 1
H0.append(sigma)
L               = [ 1 ]
H0              = H0+L
gflag           = 0
Hopt            = optimizeHP(Xt,None,yt,None,epsXt,None,H0,bounds, gflag)
sigma           = Hopt[0]
L               = Hopt[1:len(Hopt)]

# =============================================================================
# sigma = 1
# L     = np.array([1])
# =============================================================================
HPm             = np.concatenate((np.array([[sigma]]),np.array([L])),axis = 1)

""" Calculate the variances at xstar using Xtstar for every experiment 1....m"""
mat                 = kernelmatrices(x,Xt,HPm.T,epsXt)
variancevector      = np.diag(mat[0] - mat[1]@np.linalg.inv(mat[2])@mat[1].T)
y_pred              = mat[1]@np.linalg.inv(mat[2])@yt

fig, axs = plt.subplots()
fig.suptitle('Error: '+str(err)+' sigma: '+str(sigma)+' L '+str(L[0]))

plt.plot(x, freal(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.scatter(Xt, yt+epsXt.T,  label='Observations')
#plt.errorbar(Xt[:,0],yt[:,0],epsXt[0], ls='none')
plt.plot(x, y_pred, 'b-', label='Prediction')

plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - np.array([variancevector]).reshape(1000,1),
                        (y_pred + np.array([variancevector]).reshape(1000,1))[::-1]]),
         alpha=.5, fc='b', ec='None', label='variance interval')
plt.grid(True)
# =============================================================================
# axs[0].set_xlabel('$x$')
# axs[0].set_ylabel('$f(x)$')
#
# =============================================================================
