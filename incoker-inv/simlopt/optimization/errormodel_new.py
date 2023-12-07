import numpy as np
import copy
import time
from timeit import default_timer as timer
from simlopt.basicfunctions.utils.creategrid import *

def estiamteweightfactors(dy, epsphys):

    dim   = dy.shape[1]
    m     = dy.shape[2]
    delta = 1E-2
    w     = np.zeros((dy.shape[0]))

    ' Check if epsphys is just a float '
    if isinstance(epsphys, (np.floating, float)):
        SigmaLL = epsphys*np.eye(m)
        SigmaLL = np.linalg.inv(SigmaLL)
    else:
        SigmaLL = np.diagflat(epsphys)
        
    reg = delta*np.eye((dim))
    for i in range(dy.shape[0]):
        Jprime =  dy[i,:,:]       
        try:
            tmp = np.linalg.inv((Jprime@SigmaLL@Jprime.T)+reg) @ (Jprime@SigmaLL)
        except np.linalg.LinAlgError as err:
            print("Error, Matrix is not invertible.")
        w[i] = np.linalg.norm(tmp, 2)
    return w

def MCGlobalEstimate(w,var,Nall,parameterranges):
    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])
    return np.sqrt(volofparameterspace/Nall * np.dot(w,np.abs(var)))


def targetfunction(v, w, X, K, Nall, tensor, parameterranges, adaptgrad=False):

    v = v.reshape((1, -1))
    #print(v)
    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])
    CMC = (volofparameterspace/Nall)

    # Initials
    N = K.shape[0]
    m = K.shape[2]
    errorsum = 0

    # Regularization
    identitiytensor = 1E-5*np.repeat(np.identity(N)[:, :, np.newaxis],m,axis=2)
    K = K + identitiytensor

    # Interchange axis for vectorizes inverse calculation
    Kswaped = np.swapaxes(K,0,2)

    # Vectorized inverse - results in the inverse of every K - Matrix for every m
    invK  = np.linalg.inv(Kswaped)

    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))
        
    # If the gradient data is not to adapt, we just need to alter the
    # indices ":v.shape[1]"
    else:
        tmpzero = np.zeros((N,N))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)

        # Expand to a m-tensor
        tmpzero = np.repeat(tmpzero[:, :, np.newaxis],m,axis=2)
        tmpzero = np.swapaxes(tmpzero,0,2)
        invKV = np.linalg.inv(invK+tmpzero)

    # Sum along axis 2 to get to sum of all variances along the m - experiments
    tmpres  = np.sum(invKV,axis = 0)
    tmpdiag = np.diag(tmpres)

    globalerrorestimate =  CMC * np.dot(tmpdiag[:w.shape[0]],w)
    return globalerrorestimate

def gradientoftargetfunction(v, w, X, K, Nall,tensor, parameterranges,adaptgrad=False):

    v = v.reshape((1, -1))
    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])
    CMC = (volofparameterspace/Nall)

    # Initials
    N = K.shape[0]
    m = K.shape[2]
    errorgradsum = 0

    # Regularization
    identitiytensor = 1E-5*np.repeat(np.identity(N)[:, :, np.newaxis],m,axis=2)
    K = K + identitiytensor

    # Interchange axis for vectorizes inverse calculation
    Kswaped = np.swapaxes(K,0,2)

    # Vectorized inverse - results in the inverse of every K - Matrix for every m
    invK  = np.linalg.inv(Kswaped)

    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))

    # If the gradient data is not to adapt, we just need to alter the
    # indices ":v.shape[1]"
    else:
        tmpzero = np.zeros((N,N))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)

        # Expand to a m-tensor
        tmpzero = np.repeat(tmpzero[:, :, np.newaxis],m,axis=2)
        tmpzero = np.swapaxes(tmpzero,0,2)
        invKV = np.linalg.inv(invK+tmpzero)

    t0grad = time.perf_counter()
    for i in range(m):
        tmp1            = -invKV[i,:,:]@tensor@invKV[i,:,:]
        tmp2            = np.diagflat(w)@tmp1[:,:w.shape[0],:w.shape[0]]
        errorgradsum    += np.trace(tmp2,axis1=1,axis2=2)
    grad            = CMC * errorgradsum

    t1grad= time.perf_counter()
    #print("Time in grad: {}".format(t1grad-t0grad))
    return np.squeeze(grad)

def hessianoftargetfunction(v, w, X, K, Nall,tensor, parameterranges,adaptgrad=False):

    v = v.reshape((1, -1))
    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])
    CMC = (volofparameterspace/Nall)

    # Initials
    N = K.shape[0]
    m = K.shape[2]
    errorgradsum = 0

    # Hessian
    hessian = np.zeros((N,N,m))

    # Regularization
    identitiytensor = 1E-7*np.repeat(np.identity(N)[:, :, np.newaxis],m,axis=2)
    K = K + identitiytensor

    # Interchange axis for vectorizes inverse calculation
    Kswaped = np.swapaxes(K,0,2)

    # Vectorized inverse - results in the inverse of every K - Matrix for every m
    invK  = np.linalg.inv(Kswaped)

    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))

    # If the gradient data is not to adapt, we just need to alter the
    # indices ":v.shape[1]"
    else:
        tmpzero = np.zeros((N,N))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)

        # Expand to a m-tensor
        tmpzero = np.repeat(tmpzero[:, :, np.newaxis],m,axis=2)
        tmpzero = np.swapaxes(tmpzero, 0,2)
        invKV = np.linalg.inv(invK+tmpzero)

    # Reshape w for scaling
    w = np.atleast_2d((w))
    wtmp = np.repeat(w,N,axis=0).T

    t0grad = time.perf_counter()

    for i in range(m):
        for j in range(N):
            A = invKV[i,...]@tensor@invKV[i,...]
            tmp1 = invKV[i,...]@tensor[...,j]@A+A@tensor[...,j]@invKV[i,...]
            """ Produces a matrix looking like

            tmpdiag = v1v1 simga(p1) .... v1vn simga(p1)
                      ....
                      v1v1 simga(pN) .... v1vn simga(pN)

            """
            tmpdiag = np.diagonal(tmp1,axis1=0,axis2=2)
            scaledhess = tmpdiag*wtmp # Pointwise prodcut
            scaledhesssum = np.sum(scaledhess,axis = 0)
            hessian[j,:,i] = scaledhesssum
    hessian = CMC * np.sum(hessian,axis=2)

    t1grad= time.perf_counter()
    #print("Time in hess: {}".format(t1grad-t0grad))
    return hessian


def acquisitionfunction(gp,df,std,w,XGLEE,epsphys,TOLAcqui, ):
 
   'Calculate error distribution with new data for '
   acquisition = std*w
   sortedarry = -np.sort(-acquisition)
   tol = 1e-5
  
   for i in range(sortedarry.shape[0]):
       try:
            """Take the ith value"""
            tmpval =  sortedarry[i]                      #Current highest value
            oldindex = np.where((acquisition==tmpval))   #Get the corresponding point
            XC = XGLEE[oldindex]

            """ Check wether the found point is in the already known training data """
            currentindex = np.where(np.linalg.norm(gp.getX-np.squeeze(XC),2,axis=1)<tol)
            if currentindex[0].size == 0:
                return XC,i,tmpval
            else:
                continue
       except Exception:
            continue
   return np.array([]),None,None


def acquisitionfunction(gp, df, std, w, XGLEE, epsphys, TOLAcqui, generated_points_history):
    'Calculate error distribution with new data for '
    acquisition = std * w
    sortedarry = -np.sort(-acquisition)
    tol = 1e-5

    for i in range(sortedarry.shape[0]):
        try:
            """Take the ith value"""
            tmpval = sortedarry[i]  # Current highest value
            oldindex = np.where((acquisition == tmpval))  # Get the corresponding point
            XC = XGLEE[oldindex]

            """ Check wether the found point is in the already known training data """
            currentindex = np.where(np.linalg.norm(gp.getX - np.squeeze(XC), 2, axis=1) < tol)
            if currentindex[0].size == 0:
                history_index = np.where(np.linalg.norm(generated_points_history - np.squeeze(XC), 2, axis=1) < tol)
                if history_index[0].size == 0:
                    return XC, i, tmpval
            else:
                continue
        except Exception:
            continue
    return np.array([]), None, None
