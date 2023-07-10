import numpy as np
from scipy.optimize import minimize

from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from basicfunctions.reconstruction.norm import *


def functional(p, gp, SigmaLL, yreal):

    m = gp.m
    p = p.reshape((1,-1))
 
    """ GP data """
    yp = gp.predictmean(p)
    var = gp.predictvariance(p,True)
    
    if m > 1:
        SigmaVar = np.diagflat(var)
    else:
        SigmaVar = var* np.eye(m)

    """ Regularisation """
    regularization = 1E-7*np.eye(m)
   
    """ Inverse covariance matrices  """ 
    SigmaLL = np.linalg.inv(regularization+SigmaLL+SigmaVar)
    
    """ Calculating norms """
    normll = sigmanorm(yreal, yp, SigmaLL)

    return (0.5 * normll )[0,0]



def jacobianfunctional(p, gp, SigmaLL, yreal):

    m = gp.m
    p = p.reshape((1,-1))
 
    """ GP data """
    yp = gp.predictmean(p)
    var = gp.predictvariance(p)
    dy = gp.predictderivative(p)
    
    SigmaVar = var* np.eye(m)

    yp = yp.reshape((1,-1))  

    """ Tykhonov regularisation """
    regularization = 1E-7*np.eye(m)
   
    """ Inverse covariance matrices  """ 
    SigmaLL = np.linalg.inv(regularization+SigmaLL+SigmaVar)

    return np.squeeze(-dy@SigmaLL@(yreal-yp).T)

# =============================================================================
# 
#         
# def functionalwithmodel(p, fun, SigmaLL, yreal, pbar, SigmaThy):
# 
#     m = SigmaLL.shape[0]
#     p = p.reshape((1,-1))
#  
#     a = [0,2,4]
#     yp = np.zeros((3))
#     for j,m in enumerate(a):
#         yp[j] = fun["function"](p,m).reshape((-1,1))
#     yp = yp.reshape((1,-1))  
#  
#    
#     """ Inverse covariance matrices  """ 
#     SigmaLL = np.linalg.inv(SigmaLL)
# 
#     """ Calculating norms """
#     normll = sigmanorm(yreal, yp, SigmaLL)
#     return (0.5 * normll )[0,0]
# 
# 
# def jacobianfunctionalwithmodel(p, fun, SigmaLL, yreal, pbar, SigmaThy):
# 
#     m = SigmaLL.shape[0]
#     p = p.reshape((1,-1))
#  
#     a = [0,2,4]
#     yp = np.zeros((3))
#     dy = np.zeros((3,2))
#     for j,m in enumerate(a):
#         yp[j] = fun["function"](p,m).reshape((-1,1))
#         dy[j] = fun["gradient"](p,m)
#     yp = yp.reshape((1,-1))  
#     
#     """ Inverse covariance matrices  """ 
#     SigmaLL = np.linalg.inv(SigmaLL)
#     
#     #return np.squeeze(-dy@SigmaLL@(yreal-yp).T +  ((p-pbar).reshape(1, dim)@SigmaThy@np.eye(dim)).T)
#     return np.squeeze(-dy.T@SigmaLL@(yreal-yp).T)
#         
# =============================================================================
        
    