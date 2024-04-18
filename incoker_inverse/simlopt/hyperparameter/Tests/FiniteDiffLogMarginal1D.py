import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.utils.distance import *
from HPOpt.hyperparameteroptimization import *
from basicfunctions.reconstruction.deltap import *

from basicfunctions.reconstruction.functional import *
from basicfunctions.reconstruction.standarddeviation import *


def nll_fn(Xt, yt,epsXt,sigma,L):
    """ HP from H """

    """ Build K """
    N2      = Xt.shape[0]
    epsilon = np.diagflat(epsXt)

    """ Prescale data """
    Xtscaled= Xt/L
    n2sq    = np.sum(Xtscaled**2,axis=1)
    DYY     = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq) - 2*(np.dot(Xtscaled,np.transpose(Xtscaled)))
    K       = sigma**2 * np.exp(-DYY / (2.0))
    K       = K + epsilon

    # Numerically more stable implementation of Eq. (11) as described
    # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
    # 2.2, Algorithm 2.1
    L = np.linalg.cholesky(K)

    S1 = solve_triangular(L, yt, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)

    logp = np.sum(np.log(np.diagonal(L))) + \
           0.5 * yt.T@S2 + \
           0.5*N2*np.log(2*np.pi)

    return logp


def nll_fngrad(Xt, yt,Xgrad,ygrad,epsXt,sigma,L):
    """ HP from H """

    """ Build K """
    N2 = Xt.shape[0]
    N3 = Xgrad.shape[0]
    D3 = Xgrad.shape[1]

    """ Prescale data for the covaraince matrices only """
    Xtscaled    = Xt/L
    Xgradscaled = Xgrad/L

    """ Error matrices """
    epsilon     = np.diagflat(epsXt)

    n2sq = np.sum(Xtscaled**2,axis=1);
    n3sq = np.sum(Xgradscaled**2,axis=1);

    # Kernel matrix Xt Xt
    DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(Xtscaled,np.transpose(Xtscaled)))
    KYY = sigma**2 * np.exp(-DYY / 2.0)
    KYY = KYY + epsilon

    # Kernel matrix Xt Xgrad
    DXtXgrad = np.transpose(np.outer(np.ones(N3),n2sq)) + np.outer(np.ones(N2),n3sq)-2* (np.dot(Xtscaled,np.transpose(Xgradscaled)))
    kXtXgrad = sigma**2 * np.exp(-DXtXgrad / 2.0)
    KXtXgrad = np.zeros((N2,N3*D3))
    for i in range(0,N2):
        tmp = (Xt[i,:]-Xgrad)/L**2
        A = kXtXgrad[i,:]
        A = A[:,None] # Cast to coloumn vector
        tmp = np.multiply(tmp,A)
        res = np.reshape(tmp,(1,-1))
        KXtXgrad[i,:] = res

    # Kernel matrix Xgrad Xgrad
    DXgradXgrad = np.transpose(np.outer(np.ones(N3),n3sq)) + np.outer(np.ones(N3),n3sq)-2* (np.dot(Xgradscaled,np.transpose(Xgradscaled)))
    KXgXg       = sigma**2 * np.exp(-DXgradXgrad / 2.0)
    # Second derivative
    #Kfdy   = np.zeros((N3*D3,N3*D3));
    tmprow = np.array([])
    Kfdy = np.array([])
    for i in range(0,N3):
        xi = Xgrad[i,:];
        for j in range(0,N3):
            xj = Xgrad[j,:];
            diff = np.outer(((xi-xj)/(L**2)),((xi-xj)/(L**2)))
            tmp = KXgXg[i,j]*( -diff + np.diag(1/L**2))
            if j == 0:
                tmprow = tmp
            else:
                tmprow = np.concatenate((tmprow,tmp),axis=1);
        if i == 0:
            Kfdy = tmprow
        else:
            Kfdy = np.concatenate((Kfdy,tmprow),axis=0);

    Kfdy = Kfdy

    # Concatenate matrices
    K = np.concatenate((KYY,KXtXgrad),axis =1)
    K = np.concatenate((K,np.concatenate((np.transpose(KXtXgrad),Kfdy),axis =1)) ,axis =0)

    """ concatenate y and y grad """
    ytilde = np.concatenate((yt.reshape(1,-1),ygrad.reshape(1,-1)),axis = 1)

    """ Return value """
    N = N2+N3*D3

    """ Regularisation """
    gamma   = 1E-1
    n       = K.shape[0]
    reg     = gamma * np.eye(n)

    alpha = np.linalg.solve(K,ytilde.T)

    logp = 1/2*ytilde @ alpha + 1/2*np.log(np.linalg.det(K)) + N/2*np.log(2*np.pi)

    """ log marginal likelihood """
    return logp

def logmarginallikelihood_der(Xt,yt,eps,sigma,L):

    """ Jacobi matrix returns (n+nd x n) matrix without using gradient information """
    N2 = Xt.shape[0]
    D2 = Xt.shape[1]

    ytilde = yt

    epsilon = np.diagflat(eps)

    """ Prescale data """
    Xtscaled = Xt/L
    n2sq     = np.sum(Xtscaled**2,axis=1)

    """ Covariance matrix K(X,X) """
    DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq) - 2 * (np.dot(Xtscaled,np.transpose(Xtscaled)))
    K   = sigma**2 * np.exp(-DYY / 2.0) + epsilon

    """ Calculate alpha, dK/dsigma """
    invK     = np.linalg.inv(K)
    alpha    = invK@yt
    dKdsigma = 2*sigma * np.exp(-DYY / 2.0)

    """ dp / dsigma """
    dpdsigma = 0.5*np.trace((np.outer(alpha,alpha) - invK)@dKdsigma)

    """ dp / dL, dK/dL """
    dtmpdL  = np.zeros((N2,N2,D2))
    dpdL    = np.zeros((D2))
    # First derivates
    for j in range(0,N2):

        tmp = np.transpose((Xt[j,:]-Xt)**2/L**3)
        for i in range(0,D2):
            dtmpdL[j,:,i] = tmp[i,:]

    for ii in range(0,D2):
        dKdL     = K*dtmpdL[:,:,ii]
        dpdL[ii] = 0.5*np.trace((np.outer(alpha,alpha) - invK)@dKdL)

    """  concatenate """
    jacobi =  np.concatenate((np.reshape(dpdsigma,(1,1)),dpdL.reshape(D2,-1)),axis =0)

    return jacobi


dim     = 1
Xt      = np.array([[1],[2],[3]])
yt      = np.array([1,4,9])
Xgrad   = np.array([[1],[2],[3]])
ygrad   = np.array([2,4,6])
epsXt   = 1E-12*np.ones((1,3))


sigma = 1
L = np.array([1])
h = 1E-6

nllgrad  = nll_fngrad(Xt, yt,Xgrad,ygrad,epsXt,sigma,L)
nllgradh = nll_fngrad(Xt, yt,Xgrad,ygrad,epsXt,sigma+h,L)
dnlldsigma = ( nllgradh   - nllgrad ) / h

nllgradhL = nll_fngrad(Xt, yt,Xgrad,ygrad,epsXt,sigma,L+h)
dnllgraddL = ( nllgradhL   - nllgrad ) / h
# =============================================================================
#
# nll  = nll_fn(Xt, yt, epsXt,sigma,L)
# nllh = nll_fn(Xt, yt, epsXt,sigma+h,L)
# nllL1h = nll_fn(Xt, yt, epsXt,sigma,L+h)
#
# dnlldsigma = ( nllh   - nll ) / h
# dnlldL1    = ( nllL1h - nll ) / h
#
# =============================================================================

sigma = 1
L = np.array([1])
dnjacobi = logmarginallikelihood_der(Xt,yt,epsXt,sigma,L)

