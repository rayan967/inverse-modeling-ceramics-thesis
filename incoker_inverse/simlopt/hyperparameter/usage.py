import numpy as np
from scipy.optimize import minimize
from scipy.optimize import rosen, differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds

from basicfunctions.reconstruction.createExperimentalValues import *
from basicfunctions.utils.errormeasures import *
from basicfunctions.utils.createPointDistribution import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.createPointDistribution import createPD
import matplotlib.pyplot as plt

from HPOpt.Hyperparameteroptimization import *

""" Training data 1D """
# =============================================================================
# Xt      = np.array([np.linspace(0,10,4)]).T
# Xgrad   = Xt
#
# eps     = 1e-4*np.ones((1,len(Xt)))
# epsgrad = 1e-4*np.ones((1,len(Xgrad)))
#
# yt      = Xt**2
# ytg     = 2*Xgrad
#
# #logma = logmarginallikelihood(H0,Xt,Xgrad,yt,ytg,eps,epsgrad,0)
# #logmarginallikelihood_der(H0,Xt,Xgrad,yt,ytg,0)
# gflag = 0
# Hopt = optimizeHP(Xt,Xgrad,yt,ytg,eps,epsgrad,gflag,1E-8,1000,0,10)
# sigma = Hopt[0]
# L = Hopt[1:len(Hopt)][0]
# =============================================================================


""" Training data 2D """
H0 = [1, 1, 1]
def f(X):
    x = np.array([X[:,0]])
    y = np.array([X[:,1]])
    #return x.T*y.T*np.sin(np.sum(X,axis = 1))
    return x.T*np.sin(y.T)

ranges = np.array([[-2,10],[0,10]])
Xt = createPD(10, 2, "grid", ranges)

N2      = Xt.shape[0]

sigma = np.array([1])
L = np.array([2,2])
""" Prescale data """

n2sq    = np.sum(Xt**2,axis=1)
DYY     = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq) - 2*(np.dot(Xt,np.transpose(Xt)))
K       = sigma**2 * np.exp(-DYY / (2.0))
plt.imshow(K, cmap='hot', interpolation='nearest')
plt.show()
K       = K

#plt.scatter(Xt[:,0], Xt[:,1])

Xgrad = Xt
#yt  = f(Xt)

yt = np.sum(Xt**2,axis = 1)
ytg = yt
""" Startvalue """

eps     = 1e-5*np.ones((1,len(Xt)))
epsgrad = 1e-5*np.ones((1,len(Xgrad)*2))

# =============================================================================
# ineq_cons = [#'type': 'ineq', 'fun': lambda epsstar0:  Wbudget - np.sum(epsstar0*(-2E5)+500)},
#              #{'type': 'ineq', 'fun': lambda epsstar0:  (threshold-np.array([epsstar0]).reshape(-1,1))[:,0]},
#              {'type': 'ineq', 'fun': lambda x:  x}]
# optiresNM   =   minimize(logmarginallikelihood,H0,args = (Xt,Xgrad,yt,ytg,eps,epsgrad,0),
#                       method='Nelder-Mead', tol = 1e-9, options={'maxiter': 300,'disp':True})
# print(optiresNM.x)
# =============================================================================
almethod = 'Nelder-Mead'
bnds = ((0.1, 1000), (0.1, 100), (0.1, 100))
#constraints=ineq_cons,
optires   =  minimize(logmarginallikelihood,
                          H0, args=(Xt,Xgrad,yt,ytg,eps,epsgrad,0),
                          method=almethod,
                          options={'maxiter': 300, 'ftol': 1e-9, 'disp': True})
print(optires.x)

result = differential_evolution(logmarginallikelihood, bnds, args=(Xt,Xgrad,yt,ytg,eps,epsgrad,0),disp = True)
print(result.x)
"""
optires   =  minimize(normofdp,
                          eps, args=(epsXt, xstar, Xtstar, Xt, yt, m, dim),
                          method='SLSQP', jac= logmarginallikelihood_der,
                          constraints=ineq_cons, options={'maxiter': 300, 'ftol': 1e-7, 'disp': True})
"""
# =============================================================================
# logma = logmarginallikelihood(H0,Xt,Xgrad,yt,ytg,eps,epsgrad,0)
# Hopt = optimizeHP(Xt,Xgrad,yt,ytg,eps,epsgrad,1,1E-7,1000,0,100)
#
# =============================================================================
