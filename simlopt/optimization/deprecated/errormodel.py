import numpy as np
import copy
import time
from timeit import default_timer as timer

def DEBUGGRADIENDTS(X):
    N = X.shape[0]
    dim = X.shape[1]

    a = [1.0,0.9,0.8,0.7]
    dy = np.zeros((N,dim))

    for j in range(len(a)):
        for i in range(N):
            x = X[i,:]
            dy[i,:,j]  = np.array([np.cos(x[0])-a[j]*x[1]*np.sin(x[0]*x[1]), -a[j]*x[0]*np.sin(x[0]*x[1])])
    return dy


def estiamteweightfactors(dy, epsphys):

    dim = dy.shape[1]
    m = dy.shape[2]
    delta = 1E-6
    w = np.zeros((dy.shape[0]))
       
    ' Check if epsphys is just a float '
    if isinstance(epsphys, (np.floating, float)):
        SigmaLL = epsphys*np.eye(m)
        SigmaLL = np.linalg.inv(SigmaLL)
    else:
        SigmaLL = np.diagflat(epsphys)
    
    for i in range(dy.shape[0]):
        Jprime =  dy[i,:,:]   
        w[i] = np.linalg.norm((np.linalg.inv((Jprime@SigmaLL@Jprime.T+delta*np.eye((dim)))) @ (Jprime@SigmaLL)), 2)     
    return w

def MCGlobalEstimate(w,var,Nall,parameterranges):   
    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])
    return volofparameterspace/Nall * np.dot(w,np.abs(var))


def targetfunction(v, w, X, K, Nall, tensor, parameterranges, adaptgrad=False):

    v = v.reshape((1, -1))
    errorsum = 0

    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])

    'Inverse of KXX'
    invK = np.linalg.inv(K+1E-7*np.eye((K.shape[0])))

    'Inverse of KXX-1+V'
    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))
    else:
        tmpzero = np.zeros((K.shape[0],K.shape[0]))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)
        invKV = np.linalg.inv(invK+tmpzero)

    globalerrorestimate = (volofparameterspace/Nall) * np.dot(np.diag(invKV)[:w.shape[0]],w)

    return globalerrorestimate

def gradientoftargetfunction(v, w, X, K, Nall,tensor, parameterranges,adaptgrad=False):

    v = v.reshape((1, -1))
    errorgradsum = 0

    volofparameterspace = np.prod(parameterranges[:, 1] - parameterranges[:, 0])

    'Inverse of KXX'
    invK = np.linalg.inv(K+1E-7*np.eye((K.shape[0])))

    'Inverse of KXX-1+V'
    if adaptgrad:
        invKV = np.linalg.inv(invK+np.diagflat(v))
    else:
        tmpzero = np.zeros((K.shape[0],K.shape[0]))
        tmpzero[:v.shape[1],:v.shape[1]] += np.diagflat(v)
        invKV = np.linalg.inv(invK+tmpzero)

    t0grad = time.perf_counter()
    
    tmp1            = -invKV@tensor@invKV       
    tmp2            = np.diagflat(w)@tmp1[:,:w.shape[0],:w.shape[0]]
    errorgradsum    = np.trace(tmp2,axis1=1,axis2=2)
    grad            = volofparameterspace/Nall * errorgradsum

    t1grad= time.perf_counter()
    #print("Time in grad: {}".format(t1grad-t0grad))
    return np.squeeze(grad)

def hessianoftargetfunction(v, w, X, yt, hyperparameter, KXX, Nall, parameterranges, logtransform=False):

    v= v.reshape((1, -1))

    sigma= hyperparameter[0]
    Lhyper= hyperparameter[1:]
    volofparameterspace= np.prod(parameterranges[:, 1])

    errorhesssum= 0
    errorsum= 0

    'Inverse in eps for df'
    KXXdf= KXX+np.diagflat(v**(-1))
    alpha= np.linalg.solve(KXXdf, yt)

    'Inverse of KXX'
    invKXX= np.linalg.inv(KXX)

    'Inverse of KXX-1+V'
    invKV= np.linalg.inv(invKXX+np.diagflat(v))
  
    'Unit matrix from euclidean vector'
    unitmatrix= np.eye(X.shape[0])

    t0hess= time.perf_counter()
    for i, x in enumerate(X):

        ei =  unitmatrix[i, :]
        hessvar= np.zeros((Nall, Nall))

        for ii in range(Nall):

            ei_ii =  unitmatrix[ii, :]
            dvi_V = np.outer(ei_ii.T, ei_ii)

            for jj in range(ii+1):

                ei_jj =  unitmatrix[jj, :]
                dvj_V = np.outer(ei_jj.T, ei_jj)

                hessvar[ii, jj] = ei.T@(invKV@dvi_V@invKV@dvj_V@invKV+invKV@dvj_V@invKV@dvi_V@invKV)@ei

        diag= np.diagflat(np.diag(hessvar))
        hessvar += hessvar.T
        hessvar -= diag

        errorhesssum += w[i] * hessvar #w[i]**2 * hessvar

    hessian= volofparameterspace/Nall * errorhesssum

    t1hess= time.perf_counter()
    #print("Time in hess: "+str(t1hess-t0hess))
    return hessian

def acquisitionfunction(gp,df,std,w,XGLEE,epsphys,TOLAcqui,XCdummy=None):

    'Calculate error distribution with new data for '
    #acquisition = np.sqrt((var))* w.reshape((-1,1))
    acquisition = np.abs(std)* w.reshape((-1,1))

    acquidx = np.where(acquisition >= np.max(acquisition)*TOLAcqui)

    if acquidx[0].size != 0:
        XC = XGLEE[acquidx[0]]
        'Check if XC is alread in X, if so, delete points form XC'
        for i in range(gp.getX.shape[0]):
            currentindex = np.where((XC == gp.getX[i,:].tolist()).all(axis=1))
            if currentindex[0].size != 0:
                #print(" Doublicate found.")
                #print(" Delete {} from candidate points".format(gp.getX[i,:]))
                XC = np.delete(XC,(currentindex[0][0]),axis=0)
                if XCdummy is not  None:
                    XCdummy = np.vstack((XCdummy,gp.getX[i,:]))
            if XC.size == 0:
                if XCdummy is not None:
                    return XC,XCdummy
                else:
                    return XC
        if XCdummy is not None:
            return XC,XCdummy
        else:
            return XC
    else:
        print(" No changes necessary.")
        return None