from timeit import default_timer as timer

import numpy as np
from incoker_inverse.simlopt.basicfunctions.covariance.cov import *


def mse(Xt,Xgrad,yt,ygrad,hyperparameter,epsXt,epsXgrad,xs,fun,verbose = False):
    """
    

    Parameters
    ----------
    Xt : TYPE
        DESCRIPTION.
    Xgrad : TYPE
        DESCRIPTION.
    yt : TYPE
        DESCRIPTION.
    ygrad : TYPE
        DESCRIPTION.
    hyperparameter : TYPE
        DESCRIPTION.
    epsXt : TYPE
        DESCRIPTION.
    epsXgrad : TYPE
        DESCRIPTION.
    xs : TYPE
        DESCRIPTION.
    fun : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    mse : TYPE
        DESCRIPTION.

    """

    start = timer()
    error = np.zeros((xs.shape[0]))
    dim = Xt.shape[1]


    if Xgrad is None:

        mat  = kernelmatrices(xs,Xt,hyperparameter[0,:],epsXt)
        mean = mat[1]@(np.linalg.solve(mat[2],yt))

    else:
        ttilde = np.concatenate((yt,ygrad))            
        mat  = kernelmatricesgrad(xs,Xt,Xgrad,hyperparameter[0,:],epsXt,epsXgrad)
        mean = np.concatenate((mat[1], mat[3]), axis=1)@(np.linalg.solve(mat[5],ttilde))

    """ Calculate error measure """
    error = (fun(xs)-mean)**2

    """ Calculate mean error """
    mse = np.sqrt(1 / xs.shape[0]*np.sum(error))

    end = timer()

    if verbose is True:
        print("Number of data points: {}".format(Xt.shape[0]))
        if Xgrad is None:
            print("Number of gradient data points: {}".format(0))
        else:
            print("Number of gradient data points: {}".format(Xgrad.shape[0]))
        print("Mean squared error: {:g}".format(mse))
        print("Elapsed time: "+str((end - start))+" s")

    return mse



def crossvalidation(Xt,Xgrad,yt,ygrad,hyperparameter,epsXt,epsXgrad,k,verbose = False):
    """
    

    Parameters
    ----------
    Xt : TYPE
        DESCRIPTION.
    Xgrad : TYPE
        DESCRIPTION.
    yt : TYPE
        DESCRIPTION.
    ygrad : TYPE
        DESCRIPTION.
    hyperparameter : TYPE
        DESCRIPTION.
    epsXt : TYPE
        DESCRIPTION.
    epsXgrad : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    loocv : TYPE
        DESCRIPTION.

    """
    

    #Check if folding is possible
    if Xt.shape[0]%k != 0:
        print("Data couldn't be devided in equal parts. Choose other set size")
        return
    if k == 1:
        print("Data cant't be devided in one set. Choose other set size")
        return
    
    setrange=int(Xt.shape[0]/k)
    error = np.zeros((k))
    dim = Xt.shape[1]
    
    start = timer()
    
    if verbose:
        print("Data is devided into {} sets".format(int(k)))
        
    if Xgrad is None:       
        for i in range(k):
                
            """ Walking bounds """
            lower = i*setrange
            upper = (i+1)*setrange
        
            """ Validation set """
            vset = Xt[lower:upper,:].reshape((-1,dim))
            vval = yt[lower:upper].reshape((-1,1))
        
            """ Training set """
            tset = np.delete(Xt, np.s_[lower:upper], axis=0)
            tval = np.delete(yt, np.s_[lower:upper]).reshape((-1,1))
            eps = np.delete(epsXt[0,:],  np.s_[lower:upper])
        
            """ Calcualte mean """
            mat  = kernelmatrices(vset,tset,hyperparameter[0,:],eps)
            mean = mat[1]@(np.linalg.solve(mat[2],tval))
    
            """ Calculate error measure """
            error[i] = np.sqrt((1 / setrange)*np.sum( (vval-mean)**2))
    
    else:
        for i in range(k):
                
            """ Walking bounds """
            lower = i*setrange
            upper = (i+1)*setrange    
        
            """ Validation set """
            vset = Xt[lower:upper,:].reshape((-1,dim))
            vval = yt[lower:upper].reshape((-1,1))
        
            """ Training set """
            tset = np.delete(Xt, np.s_[lower:upper], axis=0)
            tsetgrad = tset
            tval = np.delete(yt, np.s_[lower:upper]).reshape((-1,1))
        
            tvalgrad = np.delete(ygrad, np.s_[lower*dim:upper*dim]).reshape((-1,1))
            
            ttilde = np.concatenate((tval,tvalgrad))
        
            eps = np.delete(epsXt[0,:],  np.s_[lower:upper])
            epsgrad = np.delete(epsXgrad[0,:],  np.s_[lower*dim:upper*dim])
        
            mat  = kernelmatricesgrad(vset,tset,tsetgrad,hyperparameter[0,:],eps,epsgrad)
            mean = np.concatenate((mat[1], mat[3]), axis=1)@(np.linalg.solve(mat[5],ttilde))
            
            """ Calculate error measure """
            error[i] = np.sqrt((1 / setrange)*np.sum( (vval-mean)**2))
    
    """ Calculate mean error """
    loocv = np.sqrt((1 / k)*np.sum(error**2))
    end = timer()
    
    if verbose:
        print("Number of data points: {}".format(Xt.shape[0]))
        if Xgrad is None:
            print("Number of gradient data points: {}".format(0))
        else:
            print("Number of gradient data points: {}".format(Xgrad.shape[0]))
        print("Mean cross validation error: {:g}".format(loocv))
        print("Elapsed time: "+str((end - start))+" s")
    
    return loocv

# =============================================================================
# """ Usage """
# """ Basic 1D data """
# testranges = np.array([[0,2]])
# Xt =  createPD(30, 1, "grid", testranges)
# yt = np.sum(Xt**2,axis=1)
# epsXt = 1E-4*np.ones((1,Xt.shape[0]))
# hyperparameter = np.array([[1,1]])
# 
# """ Basic 1D data with gradient """
# testranges = np.array([[0,2]])
# Xt =  createPD(30, 1, "grid", testranges)
# Xgrad = Xt
# yt = np.sum(Xt**2,axis=1)
# ygrad = 2*Xt
# epsXt = 1E-4*np.ones((1,Xt.shape[0]))
# epsXgrad = 1E-4*np.ones((1,Xt.shape[0]))
# hyperparameter = np.array([[1,1]])
# 
# """ Basic 2D data """
# testranges = np.array([[0,2],[0,2]])
# Xt =  createPD(4, 2, "grid", testranges)
# yt = np.sum(Xt**2,axis=1)
# epsXt = 1E-4*np.ones((1,Xt.shape[0]))
# hyperparameter = np.array([[1,1,1]])
# Xgrad = None
# verbose = True
# 
# """ Basic 2D data """
# testranges = np.array([[0,4],[0,2]])
# Xt =  createPD(20 ,2, "random", testranges)
# yt = np.sum(Xt**2,axis=1)
# ygrad = 2*Xt
# epsXt = 1E-4*np.ones((1,Xt.shape[0]))
# epsXgrad = 1E-4*np.ones((1,Xt.shape[0]*2))
# Xgrad = Xt
# verbose = True
# gp = GPR(Xt, yt.reshape((-1,1)), Xgrad, ygrad.reshape((-1,1)), epsXt, epsXgrad)
# region= ((1, 5), (0.1, 10), (0.1, 10))
# gp.optimizehyperparameter(region, "unit", False)
# hyperparameter = gp.gethyperparameter.reshape((1,-1))
# 
# k = 4 #Number of sets
# crossvalidation(Xt,Xgrad,yt,ygrad,hyperparameter,epsXt,epsXgrad,k,verbose = True)
# =============================================================================
