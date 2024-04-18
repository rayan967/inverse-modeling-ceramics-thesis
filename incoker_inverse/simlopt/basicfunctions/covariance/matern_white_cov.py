import numpy as np
import time


def kernelmatrix(X1, X2, hyperparameters):
    """
    Computes the Matérn kernel matrix for two sets of data points, X1 and X2.

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

    nu : float, optional
        The smoothness parameter of the Matérn kernel.
        Common values are 3/2 or 5/2. Default is 3/2.

    Returns
    -------
    KXY : np.array n x n
        Covariance matrix computed using the Matérn kernel.
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
    X1 = X1 / l_mat
    X2 = X2 / l_mat

    n1sq = np.sum(X1 ** 2, axis=1)
    n2sq = np.sum(X2 ** 2, axis=1)
    DXY = np.transpose(np.outer(np.ones(N2), n1sq)) + np.outer(np.ones(N1), n2sq) - 2 * (np.dot(X1, np.transpose(X2)))
    dist = np.sqrt(DXY)

    KXY = sigma ** 2 * (1 + np.sqrt(3) * dist) * np.exp(-np.sqrt(3) * dist)

    return KXY


def kernelmatrices(X1, X2, hyperparameters, eps):
    """
    Computes the Matérn kernel matrices for two sets of data points, X1 and X2.

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

    nu : float, optional
        The smoothness parameter of the Matérn kernel.
        Common values are 3/2 or 5/2. Default is 3/2.

    Returns
    -------
    KXX, KXY, KYY : np.array n x n
        Covariance matrices computed using the Matérn kernel.
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

    sigma = 1
    l_mat = hyperparameters[0:]

    # Prescale data
    X1 = X1 / l_mat
    X2 = X2 / l_mat

    # Build error matrix
    if len(eps) == 1:
        epsilon = eps * np.eye(N2)
    else:
        epsilon = np.diagflat(eps)

    n1sq = np.sum(X1 ** 2, axis=1)
    n2sq = np.sum(X2 ** 2, axis=1)

    # Compute distance matrices
    DXX = np.transpose(np.outer(np.ones(N1), n1sq)) + np.outer(np.ones(N1), n1sq) - 2 * np.dot(X1, X1.T)
    DXY = np.transpose(np.outer(np.ones(N2), n1sq)) + np.outer(np.ones(N1), n2sq) - 2 * np.dot(X1, X2.T)
    DYY = np.transpose(np.outer(np.ones(N2), n2sq)) + np.outer(np.ones(N2), n2sq) - 2 * np.dot(X2, X2.T)

    dist_XX = np.sqrt(DXX)
    dist_XY = np.sqrt(DXY)
    dist_YY = np.sqrt(DYY)

    # Compute Matérn kernel matrices

    KXX = sigma ** 2 * (1 + np.sqrt(3) * dist_XX) * np.exp(-np.sqrt(3) * dist_XX) + 1 ** 2 * np.eye(N1)
    KXY = sigma ** 2 * (1 + np.sqrt(3) * dist_XY) * np.exp(-np.sqrt(3) * dist_XY)
    KYY = sigma ** 2 * (1 + np.sqrt(5) * dist_YY) * np.exp(-np.sqrt(5) * dist_YY)

    KYY = KYY + epsilon

    return KXX, KXY, KYY