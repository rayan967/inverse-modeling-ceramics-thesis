import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import Bounds
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.reconstruction.functional import *
from basicfunctions.reconstruction.standarddeviation import *

from optimization.utils.computationalwork import*




def reconstructviamodel(fun, p0, yreal,SigmaLL, pbar, SigmaThy, region, maxiter, toliter, method, freal):
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
    dim  = 2          # number of dimensions
    allval = True
    
    method = 'Nelder-Mead'
    method = 'CG'
    method = "Newton-CG"
    method = 'L-BFGS-B'
    method = 'Powell'
    method = 'SLSQP'


    params = []
    def callback(x):
        params.append(x.copy())

    print("\n")
    print("---- Starting reconstruction ----")
    print("Problem dimension: {}".format(dim))
    print("Used method: {}".format(method))


    #options={'maxiter': maxiter,'disp': False, 'fatol':  1e-6,'xatol': 1e-6}
    #options={'maxiter': maxiter,'disp': False, 'fatol':  1e-6,'xatol': 1e-6}
    # 
    optionsSLSQP  = {'maxiter': maxiter,'disp': False, 'ftol':  1e-6}
    res  = minimize(functionalwithmodel, p0, args = (fun,SigmaLL,yreal,pbar,SigmaThy),
                    callback=callback,jac=jacobianfunctionalwithmodel,
                    bounds= region,method=method,options = optionsSLSQP)
    xmap = res.x

    if res.success == True:
        print("Reconstruction succeded in:  {} iterations".format(res.nit))
        print('Real parameter set:          {}'.format(freal))
        print('Start reconstruction at p0:  {}'.format(p0))
        print("Reconstructed parameters:    {}".format(xmap))
        print("L2 Norm of value:            {:g}".format(np.linalg.norm(freal-xmap),2))
        #print("Status:                      {}".format(res.status))
        print("Message:                     {}".format(res.message))
        return (res.x, params, False)

    elif res.success == False:
        print("Reconstruction failed          ".format(res.nit))
        print('Real parameter set:          {}'.format(freal))
        print('Start reconstruction at p0:  {}'.format(p0))
        print("Reconstructed parameters:    {}".format(xmap))
        print("L2 Norm of value:            {:g}".format(np.linalg.norm(freal-xmap),2))
        #print("Status:                      {}".format(res.status))
        print("Message:                     {}".format(res.message))
        return (res.x, params, True)
