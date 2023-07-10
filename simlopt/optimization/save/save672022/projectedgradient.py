
import numpy as np

import copy
import scipy
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from basicfunctions.covariance.cov import *
from basicfunctions.utils.creategrid import *

from basicfunctions.kaskade.kaskadeio import *

from scipy.optimize import minimize
from gpr.gaussianprocess import *

from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint


def pointwiseerror(C1, C2, df, SigmaLL, SigmaP, std, norm):

    dim = df.shape[0]

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, norm)

    L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                             np.linalg.inv(SigmaP)+1E-7*np.eye(dim))[0])  # float

    if isinstance(L, complex):
        L = L.real

    if L <= 1E-4:
        print("Warning: L is too small, set value to: "+str(1))
        L = 1

    KR = (12*C1*invnorm/L + 1/C2)
    pointwiseerror = KR *  std

    return pointwiseerror


def globalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges, logtransform = False):

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    L = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorsum = 0

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    for i, x in enumerate(X):
        'Local std'
        std = np.sqrt(unitmatrix[:, i].T@invKV@unitmatrix[:, i])

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, L)@alpha

        'Local error estimation at x \in X'
        pwee = pointwiseerror(C1, C2, df.T, SigmaLL, SigmaP, std, np.inf)
        errorsum += pwee**2

    if logtransform:
        globalerrorestimate = np.log((volofparameterspace/Nall)) + np.log(errorsum)
    else:
        globalerrorestimate = (volofparameterspace/Nall) * errorsum

    return globalerrorestimate  # 14.4


def gradientofglobalerrorestimate(v, X, yt, hyperparameter, KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges, logtransform = False):

    dim = X.shape[1]

    v = v.reshape((1, -1))

    sigma = hyperparameter[0]
    Lhyper = hyperparameter[1:]
    volofparameterspace = np.prod(parameterranges[:, 1])

    errorgradsum = 0
    errorsum = 0

    invsigma = np.linalg.inv(SigmaLL)
    invnorm = np.linalg.norm(invsigma, np.inf)

    'Inverse in eps for df'
    KXXdf = KXX+np.diagflat(v**(-1))
    alpha = np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX = np.linalg.inv(KXX)
    #invKXX[np.abs(invKXX) < 1E-6] = 0.0

    'Inverse of KXX-1+V'
    invKV = np.linalg.inv(invKXX+np.diagflat(v))
    #invKV[np.abs(invKV) < 1E-6] = 0.0

    'Unit matrix from euclidean vector'
    unitmatrix = np.eye(X.shape[0])

    for i, x in enumerate(X):

        'Euclidean unit vector'
        ei =  unitmatrix[i,:]

        'Local variance'
        var = ei.T@invKV@ei

        'Local derivative estiamte'
        KxX = KXX[i, :].reshape((1, -1))
        df = dGPR(x.reshape((1, -1)), X, KxX, Lhyper)@alpha

        L = np.min(np.linalg.eig(np.outer(df, invsigma*df) +
                   np.linalg.inv(SigmaP)+1E-4*np.eye(dim))[0])  # float

        if isinstance(L, complex):
            L = L.real
        if L <= 1E-4:
            print("Warning: L is too small, set value to: "+str(1))
            L = 1

        KR = 12/L*invnorm*C1 + (1/C2)

        gradvar = np.zeros((Nall, 1))

        for kk in range(Nall):
            ei_kk =  unitmatrix[kk,:]
            gradvar[kk, 0] = -ei.T@ invKV @ np.outer(ei_kk.T,ei_kk) @ invKV @ ei #Checked with Mathematica

        errorsum += KR * KR * var
        errorgradsum += KR * KR * gradvar

    if logtransform:
        grad = 1/errorsum * errorgradsum
    else:
        grad = volofparameterspace/Nall * errorgradsum

    return np.squeeze(grad)


def estimateconstants(gp, df):
    X = gp.getX
    C1 = np.max(np.linalg.norm(df, np.inf, axis=1))  # float

    C2 = 0
    for ii in range(X.shape[0]):
        hess = gp.predicthessian(np.array([X[ii, :]]))
        maxhess = np.linalg.norm(hess, np.inf)
        if maxhess > C2:
            C2 = maxhess

    return C1, C2


""" Inequality functions """


def totalcompwork(v, s=1):
    return np.sum(v**(s))


def totalcompworkeps(epsilon):
    return np.sum(1/2*epsilon**(-2))


def totalcompworkofalpha(alpha,v,direction,target,s=1):
    return np.sum((v+alpha*direction)**(s))-target
def jacobicompworkofalpha(alpha,v,direction,s=1):
    return np.dot(s*(v+alpha*direction)**(s-1),direction)

""" Data functions """
def linesearch(fun,jac,alpha,v,target,direction,epsilon = 1E-6,verbose=False):
    alpha = 1.0
    counter = 0

    while np.abs(fun(alpha,v,direction,target)) > epsilon:

        alpha=alpha-fun(alpha,v,direction,target)/jac(alpha,v,direction)
        vsol = v-alpha*direction

        if verbose:
            print("Iteration: "+str(counter))
            print("Current difference: "+str(np.abs(fun(alpha,v,direction,target)-target)))
            print("Current solution: "+str(vsol))
            print("Current alpha:  "+str(alpha))
            print("\n")
        counter+=1

    return vsol

def adapt(gp, N, NC, totalbudget, budgettospend,
          SigmaLL, SigmaP, parameterranges,
          TOL, TOLe, TOLFEM, TOLFILTER, loweraccuracybound, nrofnewpoints, execpath, execname,
          beta=1.4, sqpiter=1000, sqptol=1E-4):
    """

    Parameters
    ----------
    gp : Gaussian process
        Gaussian process which is getting adapted
    N : int
        Number is base data points
    NC : int
        Number of initial candidate points.
    totalbudget : float
        Total available computational budget
    budgettospend : float
        Delta W to used in every design
    SigmaLL : np.arry([[mxm]])
        Likelihood-covariance matrix
    SigmaP : np.arry([[dimxdim]])
        Parameter prior
    parameterranges : np.array([]) dimxdim
        Ranges of parameterspace
    TOL : float
        Desired tolerance for the estimated global parameter error
    TOLe : float
        Tolerance...
    loweraccuracybound : float
        lower accuracy bound for the simulated data.
    nrofnewpoints : int
        Number of new points which are set if th error isnt changing any more
    beta : float, optional
        Amplification factor for the variance. Helps not to underestimate the variance. The default is 1.2.

    Returns
    -------
    gp :  Gaussian process
        Adapted gaussian process
    errorlist : list
        List of global estimated errors.
    costlist : list
        List of costs per design.

    """
    errorlist = []
    realerrorlist = []
    costlist = []
    dim = gp.getdata[2]

    counter = 0
    totaltime = 0
    totalFEM = 0
    nosolutioncounter = 0
    itercounter = 0
    graddescenttrigger = False #Trigger if stuck in local minimum

    'Epsilon^2 at this points'
    epsilon = np.squeeze(gp.getaccuracy)

    'Solver options'
    s = 1
    method = "SLSQP"

    currentcost = totalcompworkeps(epsilon)
    costlist.append(currentcost)

    print("\n")
    print("---------------------------------- Start optimization")
    print("Number of initial points:          "+str(N))
    print("Total budget:                      "+str(totalbudget))
    print("Desired tolerance:                 "+str(TOL))
    print("Lower accuracy bound:              "+str(loweraccuracybound))
    print("Number of adaptively added points: "+str(nrofnewpoints))
    print("\n")

    while currentcost < totalbudget:

        t0design = time.perf_counter()
        print("---------------------------------- Iteration / Design: {}".format(counter))
        print("Current number of points: {} ".format(N))
        print("Current number of candidate points: {} ".format(NC))
        print("Estimate derivatives")
        df = gp.predictderivative(gp.getX, True)
        print("Estimate constants")
        C1,C2 = estimateconstants(gp,df)
        print("  C1: {:10.2f}".format(C1))
        print("  C2: {:10.2f}".format(C2))


        'Calculate KXX once, since this matrix is not changing'
        X, yt, Nall = gp.getX, gp.gety, gp.getdata[0]
        hyperparameter = gp.gethyperparameter
        KXX = kernelmatrix(X, X, hyperparameter)

        'Turn epsilon^2 into v'
        v = np.array([400.0,200.0])

        """ NEW FORMULATION STARTS HERE """

        globarerrorbefore = globalerrorestimate( v, X, yt, hyperparameter, KXX, C1, C2, Nall,
                                                 SigmaLL, SigmaP, parameterranges )
        N = Nall-NC

        print("Global error estimate before optimization:   {:1.8f}".format(globarerrorbefore))
        print("Computational cost before optimization:      {:0.0f}".format(currentcost))
        print("\n")
        print("--- Solve minimization problem")
        direction = gradientofglobalerrorestimate(v, X, yt, hyperparameter,
                                                  KXX, C1, C2, Nall, SigmaLL, SigmaP, parameterranges)
        direction*=-1
        target = 500
        vtarget = linesearch(totalcompworkofalpha,jacobicompworkofalpha,1.0, v, target, direction, 1E-4,True)