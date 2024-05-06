import numpy as np
from scipy.optimize import minimize

from simlopt.basicfunctions.covariance.cov import *
from simlopt.basicfunctions.derivative.dGPR import *

def standarddeviation(p,gp,SigmaLL,diagonalentries = True):

    """ Calculate standard deviation
    Assuming local gaussian distribution, the variances are given by the diagonal entries of
    J'' = ( f' * SigmaLL * f' + SigmaThy + geo... )^(-1)_ii
    """
    
    dim = gp.getdim
    m   = gp.m
    p   = p.reshape(1,-1)

    """ GP data """
    var = gp.predictvariance(p,True)
    if m > 1:
        SigmaVar = np.diagflat(var)
    else:
        SigmaVar = var* np.eye(m)
    
    df = gp.predictderivative(p,False)

    SigmaLL = np.linalg.inv(SigmaLL+SigmaVar)
    covariance = np.linalg.inv((df @ SigmaLL @ df.T)) 

    if diagonalentries:
        return covariance.diagonal()
    else:
        return covariance
