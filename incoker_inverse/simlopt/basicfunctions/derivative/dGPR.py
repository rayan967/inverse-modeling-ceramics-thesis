import numpy as np
from incoker_inverse.simlopt.basicfunctions.covariance.cov import *


def dGPR(X, Xt, KXY, L):
    """


    Parameters
    ----------
    X : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    Xt : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    KXY : np.array n x nd
       Covariance matrix between X and Xt

    L : np.array 1 x d
        Length scale hyperparameter

    Returns
    -------
    dfdx : TYPE
        DESCRIPTION.

    """
    X = X / L**2
    Xt = Xt / L**2

    N1 = X.shape[0]
    D1 = X.shape[1]
    D2 = Xt.shape[1]

    assert D1 == D2, "Dimensions must be equal"

  
    tmp = []
    for i in range(N1):
        tmp.append(-KXY[i, :]*np.transpose(X[i, :]-Xt))

    dfdx = np.concatenate(tmp, axis=0)

    return dfdx


def dGPRgrad(X, Xt, Xgrad, sigma, L):
    """


    Parameters
    ----------
    X : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    Xt : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    Xgrad : np.array n x d
        Matrix of gradient data.
        n - number of data points
        d - dimension of data points

    Returns
    -------
    dfdx : TYPE
        DESCRIPTION.

    """
    N1 = X.shape[0]
    N2 = Xt.shape[0]
    N3 = Xgrad.shape[0]

    D1 = X.shape[1]
    D2 = Xt.shape[1]
    D3 = Xgrad.shape[1]

    assert D1 == D2 and D2 == D3 and D1 == D3, "Dimensions must be equal"

    # Prescale data
    Xscaled = X/L
    Xtscaled = Xt/L
    Xgradscaled = Xgrad/L

    # Prescale data squared
    Xscaledsq = X/L**2
    Xtscaledsq = Xt/L**2

    # Build error matrix
    n1sq = np.sum(Xscaled**2, axis=1)
    n2sq = np.sum(Xtscaled**2, axis=1)
    n3sq = np.sum(Xgradscaled**2, axis=1)

    DXY = np.transpose(np.outer(np.ones(N2), n1sq)) + np.outer(np.ones(N1),
                                                               n2sq)-2 * (np.dot(Xscaled, np.transpose(Xtscaled)))
    KXXt = sigma**2 * np.exp(- DXY / 2.0)

    DXXgrad = np.transpose(np.outer(np.ones(N3), n1sq)) + np.outer(
        np.ones(N1), n3sq)-2 * (np.dot(Xscaled, np.transpose(Xgradscaled)))
    KXXgrad = sigma**2 * np.exp(- DXXgrad / 2.0)

    # First derivates
    tmp = []
    for i in range(0, N1):
        tmp.append(-KXXt[i, :]*np.transpose(Xscaledsq[i, :]-Xtscaledsq))
    dfdx = np.concatenate(tmp, axis=0)

    DXXgrad = np.transpose(np.outer(np.ones(N3), n1sq)) + np.outer(
        np.ones(N1), n3sq)-2 * (np.dot(Xscaled, np.transpose(Xgradscaled)))
    KXXgrad = sigma**2 * np.exp(-DXXgrad / 2.0)
    # Second derivative
    #Kfdy   = np.zeros((N3*D3,N3*D3));
    tmprow = np.array([])
    Kfdy = np.array([])
    for i in range(0, N1):
        xi = X[i, :]
        for j in range(0, N3):
            xj = Xgrad[j, :]
            diff = np.outer(((xi-xj)/(L**2)), ((xi-xj)/(L**2)))
            #tmp  = KXXgrad[i,j]*( -diff + np.diag(1/L**2))
            tmp = KXXgrad[i, j]*(-diff + np.diagflat(1/L**2))
            if j == 0:
                tmprow = tmp
            else:
                tmprow = np.concatenate((tmprow, tmp), axis=1)
        if i == 0:
            Kfdy = tmprow
        else:
            Kfdy = np.concatenate((Kfdy, tmprow), axis=0)

    # Concatenate matrices
    dfgraddx = np.concatenate((dfdx, Kfdy), axis=1)

    return dfgraddx
