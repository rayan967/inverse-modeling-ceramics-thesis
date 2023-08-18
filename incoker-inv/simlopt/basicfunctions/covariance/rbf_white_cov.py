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
    KXY = sigma**2 * np.exp(-DXY / 2.0*4)

    return KXY


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
    KXX = sigma**2 * np.exp(-DXX / 2.0*4) + 1 ** 2 * np.eye(N1)

    DXY = np.transpose(np.outer(np.ones(N2),n1sq)) + np.outer(np.ones(N1),n2sq)-2* (np.dot(X1,np.transpose(X2)))
    #DXY[np.abs(DXY) < 1E-6] = 0.0
    KXY = sigma**2 * np.exp(-DXY / 2.0*4)

    DYY = np.transpose(np.outer(np.ones(N2),n2sq)) + np.outer(np.ones(N2),n2sq)-2* (np.dot(X2,np.transpose(X2)))
    #DYY[np.abs(DYY) < 1E-6] =0.0
    KYY = sigma**2 * np.exp(-DYY / 2.0*4)
    KYY = KYY + epsilon

    #ret = []
    #ret.extend([KXX,KXY,KYY])

    return KXX,KXY,KYY

