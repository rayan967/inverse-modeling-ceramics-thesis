import numpy as np


from scipy.optimize import *
from scipy.optimize import minimize

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.creategrid import *

from hyperparameter.utils.logmarginal import *
from hyperparameter.utils.setstartvalues import*

def optimizehyperparameters(Xt, Xgrad, yt, ygrad, epsXt, epsXgrad, region, startvaluescheme):
    """

    Parameters
    ----------
    Xt : np.array n x d x m
        DESCRIPTION.
    Xgrad : np.array n x d x m
        DESCRIPTION.
    yt : np.array n x m
        DESCRIPTION.
    ygrad : np.array n x d x m
        DESCRIPTION.
    epsXt : m x n
        DESCRIPTION.
    epsXgrad : m x n
        DESCRIPTION.
    region : tuple
        DESCRIPTION.
    startvaluescheme : string
        DESCRIPTION.

    Returns
    -------
    HPm : np.array 1 x d +1
        DESCRIPTION.

    """

    dim = Xt.shape[1]

    if Xgrad is None:
        print("Performing hyperparameter optimization without gradient information")
        adaptgrad = False
    else:
        print("Performing hyperparameter optimization with gradient information")
        adaptgrad = True

    if startvaluescheme == "unit":
        print("Startvalues set to unit values [1,...,1]")
        L = [1 for i in range(dim)]
    if startvaluescheme == "mean":
        print("Startvalues set by calculating the mean of every feature")
        L = findL(Xt)
    print(" Startvalue(s): "+np.array2string(np.array(L), precision=3, separator=', ',suppress_small=True))
    
    Lopt = minlogmll(Xt, Xgrad, yt, ygrad, epsXt, epsXgrad, L, region, adaptgrad , verbose = False)

    return Lopt


def optimizehyperparametersmultistart(Xt, Xgrad, yt, ygrad, epsXt, epsXgrad, region):
    """

    Parameters
    ----------
    Xt : np.array n x d x m
        DESCRIPTION.
    Xgrad : np.array n x d x m
        DESCRIPTION.
    yt : np.array n x m
        DESCRIPTION.
    ygrad : np.array n x d x m
        DESCRIPTION.
    epsXt : m x n
        DESCRIPTION.
    epsXgrad : m x n
        DESCRIPTION.
    region : tuple
        DESCRIPTION.
    startvaluescheme : string
        DESCRIPTION.

    Returns
    -------
    HPm : np.array m x d +1
        DESCRIPTION.

    """

    dim = Xt.shape[1]
    data = []

    'Create startvalue hypercube'
    loggrid = 10.**(-np.arange(-1, 2))
    
    allG = []
    for i in range(dim):
        allG.append(loggrid)# Create linspaces
        out = np.meshgrid(*allG)
    outtmp = []
    for j in range(dim):
        outtmp.append( out[j].reshape(-1))
    grid = np.vstack([*outtmp]).T

    "Calculate everything for any given start value"
    counter = 1
    for kk in range(grid.shape[0]):
        
        L = list(grid[kk,:])
        print("Iteration: "+str(counter))
        print(" Startvalue(s): "+np.array2string(np.array(L), precision=3, separator=', ',suppress_small=True))

        if Xgrad is None:
            #print("Performing hyperparameter optimization without gradient information")
            adaptgrad = False
        else:
            #print("Performing hyperparameter optimization with gradient information")
            adaptgrad = True
                
        Lopt = minlogmll(Xt, Xgrad, yt, ygrad, epsXt, epsXgrad, L, region, adaptgrad ,1E-4,500, False, True)    
            
        if Lopt[1] is not None:
            data.append(Lopt)
        else:
            print("No solution was found, result was not saved.")
        counter += 1
        
    "Compare and return"
    currlogml = 0
    currhp = np.zeros((m, dim + 1))
    for jj in range(len(data)):
        if data[jj] is not None:
            if  np.abs(data[jj][1]) > currlogml:
                currlogml = data[jj][1]
                HPm = data[jj][0]
    print("Chosen hyperparameter: "+ str(HPm)+ " with log marginal ll: " + str(currlogml))
    return HPm

def minlogmll(Xt, Xgrad, yt, ygrad,
              eps, epsgrad,
              startvalues, region,
              gflag = 0,
              toliter = 1E-4, maxiter = 10000,verbose=False, returnoptimalfuncvalue = False,verboseminimizer=True):
    """

    Parameters
    ----------
    Xt : np.array n x d
        Training data

    Xgrad : np.array n x d
        Gradient data

    yt : np.array 1 x d
        Values at Xt

    ygrad : np.array n * d x 1
        Values at Xgrad

    eps : TYPE
        DESCRIPTION.

    epsgrad : TYPE
        DESCRIPTION.

    startvalues : np.array, optional
        Startvalues for the iteration.

    region : tuple
        bounds for the hyperparameter

    gflag : TYPE, optional
        DESCRIPTION. The default is 0.

    toliter : TYPE, optional
        DESCRIPTION. The default is 1E-8.

    maxiter : TYPE, optional
        DESCRIPTION. The default is 300.

    returnoptimalfuncvalue : bool
        DESCRIPTION. The default is False.

    Returns
    -------
    np.array(dim+1) with the optimizied hyperparameters

    """
    
    dim = Xt.shape[1]

    params = []
    def callback(x):
        params.append(x.copy())

    almethod = 'trust-constr'

    if verbose:
        print("Algorithm: {}".format(almethod))
        
    try:        
        
        def calculatecondition(L,Xt,Xgrad,eps,epsgrad):
            if Xgrad is not None:
                K = kernelmatrixsgrad(Xt,Xgrad,L,eps,epsgrad)
            else:
                _,_,K = kernelmatrices(Xt,Xt,L,eps)
            return np.linalg.cond(K, 'fro')
                
        condition_constraint = NonlinearConstraint(lambda L: calculatecondition(L,Xt,Xgrad,eps,epsgrad), 0, 1E7)
        #constraints=[condition_constraint],
        optires = minimize(logmarginallikelihood,
                            startvalues,args = (Xt,Xgrad,yt,ygrad,eps,epsgrad,gflag),
                            method=almethod, jac=logmarginallikelihood_der, hess = BFGS(),
                            bounds = region,
                            tol = toliter, options={'maxiter': maxiter,'disp':False,'verbose': 0,'xtol': 1E-3, 'gtol': 1E-3})
    
    except np.linalg.LinAlgError as e:
        print("Optimization failed")
        print(" "+ str(e))
        return None
    
    if optires.success == True:
        if verbose:
            print("Hyperparameter optimization succeded in {} iterations".format(optires.nit))
            print(" Optimized hyperparameter: "+np.array2string(np.abs(optires.x), precision=2, separator=',',suppress_small=True))
            #print(" Condition of covariance matrix: {:1.0f}".format(calculatecondition(optires.x,Xt,Xgrad,eps,epsgrad)))
        if returnoptimalfuncvalue is True:
            'Returns also the optimizhed function value. This is used within multistart methods'
            print(" Log margnal ll value: {}".format(np.squeeze(optires.fun)))
            return np.abs(optires.x),optires.fun
        else:
            return np.abs(optires.x)
    else:
        return np.ones(dim)
