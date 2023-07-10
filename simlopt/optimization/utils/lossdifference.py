import numpy as np

from basicfunctions.utils.creategrid import *
from basicfunctions.covariance.cov import *
from basicfunctions.derivative.dGPR import *
from HPOpt.hyperparameteroptimization import *

from HPOpt.utils.setstartvalues import *
from basicfunctions.utils.creategrid import *
from Optimization.utils.computationalwork import *
from Reconstruction.utils.perror import *


' Lossfunction and its derivate '
def lossdifference(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys):
    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[0,idx] = current_eps**2

    """ Calculate dp with new eps at x """
    matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
    metaerror = matrices[0] - (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)
        
    L = HPm[0, 1:]
    alpha = np.linalg.inv(matrices[2]) @ ytextended
    df = dGPR(x, Xtextended, matrices[1], L)@alpha
    
    

    #SigmaInv = np.diagflat(1/metaerror)
    
    dp = parametererror(df,metaerror,epsphys,m,dim)
# =============================================================================
#     print("df: {}".format(df))
#     print("metaerror: {}".format(metaerror))
#     print("dp: {}".format(np.linalg.norm(dp,2)))
# =============================================================================
# =============================================================================
#     if dim != 1:
#         A = metaerror * 1/(np.dot(df.T,df))
#         B = df.T*SigmaInv*metaerror.T
#         dp = -A*B
#         
#         return np.linalg.norm(dp,2)**2
#     else:  
#         A = np.linalg.inv(df.T@SigmaInv@df)
#         B = df.T@SigmaInv@metaerror.T
#         dp = -A@B
# =============================================================================
    return np.linalg.norm(np.abs(dp),2)**2
#
def dlossdifference(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys):
    """ Calcualte eps(W) """
    current_eps = epsofw(w0, dim)

    """ Set new eps-training data """
    epsXtextended[0,idx] = current_eps**2

    """ Calcualte dp with new eps at x """
    matrices = kernelmatrices(x, Xtextended, HPm[0, :], epsXtextended)
    metaerror = matrices[0] - \
        (matrices[1]@np.linalg.inv(matrices[2])@matrices[1].T)

    L = HPm[0, 1:]
    alpha = np.linalg.inv(matrices[2]) @ ytextended
    df = dGPR(x, Xtextended, matrices[1], L)@alpha

    SigmaInv = np.diagflat(1/(metaerror+epsphys) )
    if dim != 1:
        A = (metaerror+epsphys) * 1/(np.dot(df.T,df))
        B = df.T*SigmaInv*(metaerror+epsphys).T
        dp = -A*B    
    else:  
        A = np.linalg.inv(df.T@SigmaInv@df)
        B = df.T@SigmaInv@(metaerror+epsphys).T
        dp = -A@B
    
    """ Calculate dpdw with new eps at x """

    dfdepsj = np.zeros((m, dim))
    dEdepsi = np.zeros((Xtextended.shape[0], Xtextended.shape[0]))

    eps = epsXtextended[0,:]
    epsj = epsXtextended[0,idx]
    dEdepsi[idx, idx] = 2*np.sqrt(epsj)

    matricesdf = kernelmatrices(x, Xtextended, HPm[0, :], eps)
    L = HPm[0, 1:]

    invK = np.linalg.inv(matricesdf[2])

    if dim != 1:
        # (invK @ dEdepsi @ invK)mit Mathematica getestet
        dfdepsj[0, :] = (-dGPR(x, Xtextended, matricesdf[1],L) @ (invK @ dEdepsi @ invK) @ ytextended).reshape((1,-1))
    else:
        dfdepsj[0, :] = -dGPR(x, Xtextended, matricesdf[1],L) @ (invK @ dEdepsi @ invK) @ ytextended
    #dfdepsj[i,:]  mit Mathematica gecheckt

    """ Ableitung var nach deps """
    dvardepsj = np.zeros((m, 1))
    dvardepsj[0, :] = matricesdf[1] @ invK @ dEdepsi @ invK @ matricesdf[1].T
    #ddvardepsj[i,:]  mit Mathematica gecheckt

    """ Ableitung SigmaInv nach deps """
    dSigmaInvdepsi = np.zeros((m, m))
    dSigmaInvdepsi[0, 0] = (matricesdf[1]@invK @ dEdepsi @ invK@matricesdf[1].T)[0, 0]
    
    if dim != 1:
        dAdepsj = -A@(dfdepsj@(SigmaInv@df.T).T - df.T@(SigmaInv@dSigmaInvdepsi@SigmaInv@df.T).T + df.T @ (SigmaInv@dfdepsj).T)@A 
        dBdepsj = dfdepsj.T@SigmaInv@(metaerror+epsphys).T - df@(SigmaInv@dSigmaInvdepsi@SigmaInv@(metaerror+epsphys).T) + df@SigmaInv@dvardepsj        
        ddpdepsj = -((dAdepsj@B) + (A@dBdepsj.T))
        dndpdeps = 2*np.dot((dp), ddpdepsj.T)    
   
    else:
    
        dAdepsj = -A@(dfdepsj.T@SigmaInv@df - df.T@SigmaInv @
                      dSigmaInvdepsi@SigmaInv @ df + df.T @ SigmaInv@dfdepsj)@A
        dBdepsj = dfdepsj.T@SigmaInv@(metaerror+epsphys).T - \
            df.T@SigmaInv@dSigmaInvdepsi@SigmaInv@(metaerror+epsphys).T + df.T@SigmaInv@dvardepsj
        ddpdepsj = -((dAdepsj@B) + (A@dBdepsj))
        dndpdeps = 2*np.dot((dp).T, ddpdepsj)

    ' deps / dw '
    w = W(epsj, dim)
    depsdW = deps(w,dim)
    
    ' Compose everything'
    dndpdW = dndpdeps * depsdW 

    ' return '
    return -dndpdW[0,0]


def hlossdifference(w0, idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys):

    res = np.zeros(2)
    h = np.array([0,1E-7])    
        
    for i in range(2):
        res[i] = dlossdifference(w0+h[i], idx, x, Xtextended, ytextended, epsXtextended, HPm , m, dim, epsphys)
    
    ' return '
    return (res[1]-res[0]) / h[1]