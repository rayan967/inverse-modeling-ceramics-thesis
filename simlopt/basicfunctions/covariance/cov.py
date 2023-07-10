import numpy as np
import time

def kernelmatrix(X1,X2,hyperparameters):
    """


    Parameters
    ----------
    X1 : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    X2 : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    hyperparameters : np.array 1x(1+d)
        Array of hyperparamters where sigma is the first entry in the list
        hyperparameters = [sigma , L1 , L2 ,... , Ld ]

    eps : np.array 1xd
        Vector of data point errors
        eps = [eps_1^2,.... , eps_d^2]

    Returns
    -------
    ret : List of covariance matrices


    """
    N1 = X1.shape[0]
    D1 = X1.shape[1]

    N2 = X2.shape[0]
    D2 = X2.shape[1]

    assert D1 == D2, "Dimensions must be equal"

    # Preallocation of covariance matrices
    KXY = np.zeros((N1, N2))

    sigma = 1
    l_mat = hyperparameters

    # Prescale data
    X1 = X1/l_mat
    X2 = X2/l_mat

    n1sq = np.sum(X1**2,axis=1);
    n2sq = np.sum(X2**2,axis=1);

    DXY = np.transpose(np.outer(np.ones(N2),n1sq)) + np.outer(np.ones(N1),n2sq)-2* (np.dot(X1,np.transpose(X2)))
    #DXY[np.abs(DXY) < 1E-8] = 0
    KXY = sigma**2 * np.exp(-DXY / 2.0)

    return KXY

def kernelmatrixsgrad(Xt,Xgrad,hyperparameters,eps,epsgrad):
    """
    s

    Parameters
    ----------
    X : np.array n x d
        Matrix of evalutation data

    Xt : np.array n x d
        Matrix of trainig data

    Xgrad : np.array n x d
        Matrix of gradient data

    hyperparameters : np.array 1x(1+d)
        Array of hyperparamters where sigma is the first entry in the list
        hyperparameters = [sigma , L1 , L2 ,... , Ld ]

    eps : np.array 1xd
        Vector of data point errors
        eps = [eps_1^2,.... , eps_d^2]

    epsgrad : np.array 1xd*n
        epsilongrad = [ epsilon_11 , epsilon_12, epsilon_1d , ...,epsilon_n1 , epsilon_n2, epsilon_nd]

    Returns
    -------
    ret : TYPE
        DESCRIPTION.

    """
    t0 = time.perf_counter()


    N2 = Xt.shape[0]
    D2 = Xt.shape[1]

    N3 = Xgrad.shape[0]
    D3 = Xgrad.shape[1]

    sigma = 1
    L = hyperparameters[0:]

    # Prescale data
    Xtscaled    = Xt/L
    Xgradscaled = Xgrad/L

    # Build error matrices

    if len(eps) == 1:
        epsilon = eps*np.eye(N2)
    else:
        epsilon = np.diagflat(eps)

    epsilongrad = np.diagflat(epsgrad)

    # Build kernel matrices
    n2sq = np.sum(Xtscaled**2,axis=1);
    n3sq = np.sum(Xgradscaled**2,axis=1);

    # Kernel matrix Xt Xt
    DXtXt = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(Xtscaled,np.transpose(Xtscaled)))
    KXtXt = sigma**2 * np.exp(-DXtXt / 2.0)
    KXtXt = KXtXt

    """ ---- Derivative matrices ---- """

    # Kernel matrix Xt Xgrad
    DXtXgrad = np.transpose(np.outer(np.ones(N3),n2sq)) + np.outer(np.ones(N2),n3sq)-2* (np.dot(Xtscaled,np.transpose(Xgradscaled)))
    DXtXgrad[np.abs(DXtXgrad) < 1E-6] = 0.0
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
    DXgradXgrad[np.abs(DXgradXgrad) < 1E-6] = 0.0
    KXgXg = sigma**2 * np.exp(-DXgradXgrad / 2.0)
    # Second derivative
    #Kfdy   = np.zeros((N3*D3,N3*D3));
    tmprow = np.array([])
    Kfdy = np.array([])
    for i in range(0,N3):
        xi = Xgrad[i,:]
        for j in range(0,N3):
            xj = Xgrad[j,:]
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
    # Concatenate matrices
    K = np.concatenate((KXtXt,KXtXgrad),axis =1)
    K = np.concatenate((K,np.concatenate((np.transpose(KXtXgrad),Kfdy),axis =1)) ,axis =0)
    return K


def kernelmatrices(X1,X2,hyperparameters,eps):
    """


    Parameters
    ----------
    X1 : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    X2 : np.array n x d
        Matrix of data points.
        n - number of data points
        d - dimension of data points

    hyperparameters : np.array 1x(1+d)
        Array of hyperparamters where sigma is the first entry in the list
        hyperparameters = [sigma , L1 , L2 ,... , Ld ]

    eps : np.array 1xd
        Vector of data point errors
        eps = [eps_1^2,.... , eps_d^2]

    Returns
    -------
    ret : List of covariance matrices


    """
    N1 = X1.shape[0]
    D1 = X1.shape[1]

    N2 = X2.shape[0]
    D2 = X2.shape[1]

    assert D1 == D2, "Dimensions must be equal"

    # Preallocation of covariance matrices
    KXX = np.zeros((N1, N1))
    KXY = np.zeros((N1, N2))
    KYY = np.zeros((N2, N2))

# =============================================================================
#     sigma = hyperparameters[0]
#     l_mat = hyperparameters[1:1+D2]
# =============================================================================
    sigma = 1
    l_mat = hyperparameters[0:]

    # Prescale data
    X1 = X1/l_mat
    X2 = X2/l_mat

    # Build error matrix
    if len(eps) == 1:
        epsilon = eps*np.eye(N2)
    else:
        epsilon = np.diagflat(eps)

    n1sq = np.sum(X1**2,axis=1);
    n2sq = np.sum(X2**2,axis=1);

    DXX = np.transpose(np.outer(np.ones(N1),n1sq)) + np.outer(np.ones(N1),n1sq)-2* (np.dot(X1,np.transpose(X1)))
    #DXX[np.abs(DXX) < 1E-6] = 0.0
    KXX = sigma**2 * np.exp(-DXX / 2.0)

    DXY = np.transpose(np.outer(np.ones(N2),n1sq)) + np.outer(np.ones(N1),n2sq)-2* (np.dot(X1,np.transpose(X2)))
    #DXY[np.abs(DXY) < 1E-6] = 0.0
    KXY = sigma**2 * np.exp(-DXY / 2.0)

    DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(X2,np.transpose(X2)))
    #DYY[np.abs(DYY) < 1E-6] =0.0
    KYY = sigma**2 * np.exp(-DYY / 2.0)
    KYY = KYY + epsilon

    #ret = []
    #ret.extend([KXX,KXY,KYY])

    return KXX,KXY,KYY


def kernelmatricesgrad(X,Xt,Xgrad,hyperparameters,eps,epsgrad):
    """
    s

    Parameters
    ----------
    X : np.array n x d
        Matrix of evalutation data

    Xt : np.array n x d
        Matrix of trainig data

    Xgrad : np.array n x d
        Matrix of gradient data

    hyperparameters : np.array 1x(1+d)
        Array of hyperparamters where sigma is the first entry in the list
        hyperparameters = [sigma , L1 , L2 ,... , Ld ]

    eps : np.array 1xd
        Vector of data point errors
        eps = [eps_1^2,.... , eps_d^2]

    epsgrad : np.array 1xd*n
        epsilongrad = [ epsilon_11 , epsilon_12, epsilon_1d , ...,epsilon_n1 , epsilon_n2, epsilon_nd]

    Returns
    -------
    ret : TYPE
        DESCRIPTION.

    """
    t0 = time.perf_counter()
    N1 = X.shape[0]
    D1 = X.shape[1]

    N2 = Xt.shape[0]
    D2 = Xt.shape[1]

    N3 = Xgrad.shape[0]
    D3 = Xgrad.shape[1]

    assert D1 == D2 and D2 == D3 and D1 == D3, "Dimensions must be equal"

# =============================================================================
#     sigma = hyperparameters[0]
#     L = hyperparameters[1:1+D2]
# =============================================================================
    sigma = 1
    L = hyperparameters[0:]

    # Prescale data
    Xscaled     = X/L
    Xtscaled    = Xt/L
    Xgradscaled = Xgrad/L

    # Build error matrices

    if len(eps) == 1:
        epsilon = eps*np.eye(N2)
    else:
        epsilon = np.diagflat(eps)

    epsilongrad = np.diagflat(epsgrad)

    # Build kernel matrices
    n1sq = np.sum(Xscaled**2,axis=1);
    n2sq = np.sum(Xtscaled**2,axis=1);
    n3sq = np.sum(Xgradscaled**2,axis=1);

    # Kernel matrix X X
    DXX = np.transpose(np.outer(np.ones(N1),n1sq)) + np.outer(np.ones(N1),n1sq)-2* (np.dot(Xscaled,np.transpose(Xscaled)))
    KXX = sigma**2 * np.exp(-DXX / 2.0)

    # Kernel matrix X Xt
    DXXt = np.transpose(np.outer(np.ones(N2),n1sq)) + np.outer(np.ones(N1),n2sq)-2* (np.dot(Xscaled,np.transpose(Xtscaled)))
    KXXt = sigma**2 * np.exp(-DXXt / 2.0)

    # Kernel matrix Xt Xt
    DXtXt = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(Xtscaled,np.transpose(Xtscaled)))
    KXtXt = sigma**2 * np.exp(-DXtXt / 2.0)
    KXtXt = KXtXt + epsilon

    """ ---- Derivative matrices ---- """

    # Kernel matrix X Xgrad
    DXXgrad = np.transpose(np.outer(np.ones(N3),n1sq)) + np.outer(np.ones(N1),n3sq)-2* (np.dot(Xscaled,np.transpose(Xgradscaled)))
    #DXXgrad[np.abs(DXXgrad) < 1E-6] = 0.0
    kXXgrad = sigma**2 * np.exp(-DXXgrad / 2.0)
    KXXgrad = np.zeros((N1,N3*D3))
    for i in range(0,N1):
        tmp = (X[i,:]-Xgrad)/L**2
        A = kXXgrad[i,:]
        A = A[:,None] # Cast to column vector
        tmp = np.multiply(tmp,A)
        res = np.reshape(tmp,(1,-1))
        KXXgrad[i,:] = res

    # Kernel matrix Xt Xgrad
    DXtXgrad = np.transpose(np.outer(np.ones(N3),n2sq)) + np.outer(np.ones(N2),n3sq)-2* (np.dot(Xtscaled,np.transpose(Xgradscaled)))
    #DXtXgrad[np.abs(DXtXgrad) < 1E-6] = 0.0
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
    #DXgradXgrad[np.abs(DXgradXgrad) < 1E-6] = 0.0
    KXgXg = sigma**2 * np.exp(-DXgradXgrad / 2.0)
    # Second derivative
    #Kfdy   = np.zeros((N3*D3,N3*D3));
    tmprow = np.array([])
    Kfdy = np.array([])
    for i in range(0,N3):
        xi = Xgrad[i,:]
        for j in range(0,N3):
            xj = Xgrad[j,:]
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

    Kfdy = Kfdy + epsilongrad

    # Concatenate matrices
    K = np.concatenate((KXtXt,KXtXgrad),axis =1)
    K = np.concatenate((K,np.concatenate((np.transpose(KXtXgrad),Kfdy),axis =1)) ,axis =0)

    ret = []
    ret.extend([KXX,KXXt,KXtXt,KXXgrad,KXtXgrad,K])
    t1 = time.perf_counter()
    #print("Cov time: {}".format(t1-t0))
    return KXX,KXXt,KXtXt,KXXgrad,KXtXgrad,K


def kernelmatrixfd(X, Xgrad,hyperparameters):
    
    N1 = X.shape[0]
    D1 = X.shape[1]

    N3 = Xgrad.shape[0]
    D3 = Xgrad.shape[1]

# =============================================================================
#     sigma = hyperparameters[0]
#     L = hyperparameters[1:1+D1]
# =============================================================================

    sigma = 1
    L = hyperparameters[0:1+D1]

    # Prescale data
    Xscaled     = X/L
   
    Xgradscaled = Xgrad/L

    # Build kernel matrices
    n1sq = np.sum(Xscaled**2,axis=1);
    n3sq = np.sum(Xgradscaled**2,axis=1);

    DXXgrad = np.transpose(np.outer(np.ones(N3),n1sq)) + np.outer(np.ones(N1),n3sq)-2* (np.dot(Xscaled,np.transpose(Xgradscaled)))
    kXXgrad = sigma**2 * np.exp(-DXXgrad / 2.0)
    KXXgrad = np.zeros((N1,N3*D3))
    for i in range(0,N1):
        tmp = (X[i,:]-Xgrad)/L**2
        A = kXXgrad[i,:]
        A = A[:,None] # Cast to column vector
        tmp = np.multiply(tmp,A)
        res = np.reshape(tmp,(1,-1))
        KXXgrad[i,:] = res
        
    return KXXgrad


def kernelmatrixsd(x,hyperparameters):

    N3 = x.shape[0]
    D3 = x.shape[1]

    sigma = hyperparameters[0]
    L = hyperparameters[1:1+D3]

    # Prescale data
    Xgradscaled = x/L
    n3sq = np.sum(Xgradscaled**2,axis=1);
    
     # Kernel matrix Xgrad Xgrad
    DXgradXgrad = np.transpose(np.outer(np.ones(N3),n3sq)) + np.outer(np.ones(N3),n3sq)-2* (np.dot(Xgradscaled,np.transpose(Xgradscaled)))
    KXgXg = sigma**2 * np.exp(-DXgradXgrad / 2.0)

    tmprow = np.array([])
    Kfdy = np.array([])
    for i in range(0,N3):
        xi = x[i,:]
        for j in range(0,N3):
            xj = x[j,:]
            diff = np.outer(((xi-xj)/(L**2)),((xi-xj)/(L**2)))
            tmp = KXgXg[i,j]*( -diff + np.diag(1/L**2))
            if j == 0:
                tmprow = tmp
            else:
                tmprow = np.concatenate((tmprow,tmp),axis=1);
        if i == 0:
            Kfdy = tmprow
        else:
            Kfdy = np.concatenate((Kfdy,tmprow),axis=0)
    
    return Kfdy