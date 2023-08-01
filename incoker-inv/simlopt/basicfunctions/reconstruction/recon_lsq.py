import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import least_squares
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.reconstruction.functional import *
from basicfunctions.reconstruction.standarddeviation import *

from optimization.utils.computationalwork import*


def reconstruct(gp, p0, yreal,SigmaLL, pbar, SigmaThy, region, maxiter, toliter, method, freal):
    """
    Parameters
    ----------
    p0 : np.array 1 x d
        Start value for the reconstruction.

    res : tuple
        Tuple containing
        (Xt,yt,Xgrad,ygrad,hyperparameters,epsXt,epsXgrad)
        from the offline training

    yreal : np.array m x 1
        Vector with "real" experimental values.

    maxiter : int
        Maximum of iterations for the minimizer.

    toliter : float
        Minimum tolerance for the minimizer.

    Returns
    -------
    p : np.array 1xd
        Reconstructed geometrical paremeters

    """
    n    = (gp.getX).shape[0] # number of training points
    dim  = gp.getdim          # number of dimensions

    print("\n")
    print("---- Starting reconstruction ----")
    print("Problem dimension: {}".format(dim))
    print('Number of training points {}'.format(n))
    print("Used method: {}".format(str(method)))

    if gp.getXgrad is None:
        ngrad = 0
    else:
        ngrad = gp.getXgrad.shape[0]
    print('    Number of gradient enhanced training points {}'.format(ngrad))

    options={'maxiter': maxiter,'disp': True, 'return_all': True, 'fatol': toliter,'xatol': 1e-8}
    res     = GaussNewton(p0,gp,yreal,SigmaLL, pbar, SigmaThy, maxiter, toliter)
    
    xmap = res[0]
    std = standarddeviation(xmap, gp, SigmaLL, SigmaThy)

    nit = len(res[1])

    if res[2] == True:
        print("Reconstruction succeded in:  {} iterations".format(nit))
        print('Real parameter set:          {}'.format(freal))
        print('Start reconstruction at p0:  {}'.format(p0))
        print("Reconstructed parameters:    {}".format(xmap))
        print("L2 Norm of value:            {:g}".format(np.linalg.norm(freal-xmap),2))
        print("Standard deviation:          {}".format(std))
        return (res[0], std, res[1])

    elif  res[2] == False:
        print("Reconstruction failed          ".format(nit))
        print('Real parameter set:          {}'.format(freal))
        print('Start reconstruction at p0:  {}'.format(p0))
        print("Reconstructed parameters:    {}".format(xmap))
        print("L2 Norm of value:            {:g}".format(np.linalg.norm(freal-xmap),2))
        print("Standard deviation:          {}".format(std))
        return (res[0], std, res[1])


def GaussNewton(p0,gp,yreal,SigmaLL, pbar, SigmaThy, maxiter, TOL):
    
    p = np.atleast_2d(p0)
    delta = 1E-6
    iterlist = []
    
    for i in range(maxiter):
        
        'Mean at p'
        mean = gp.predictmean(p)
        
        'Derivative at p'
        yd = gp.predictderivative(p)
        
        'Weights'
        W = np.linalg.inv(SigmaLL)
        Wprior = np.linalg.inv(SigmaThy)
        
        'Calcualte residual'
        R = np.concatenate((W@(yreal-mean).T, Wprior@(p-pbar).T), axis=0)

        'Jprime'
        Jprime = np.concatenate((W@yd.T,Wprior),axis=0)
        
        'Calculate dp'       
        deltap = np.linalg.inv(Jprime.T@Jprime+delta*np.eye((2)))@Jprime.T@R
        
        norm = np.linalg.norm(deltap,2)
        iterlist.append(norm)
        
        if norm < TOL:
            return p,iterlist,True
        
        p = p + deltap.T
    else:
        return p,iterlist,False